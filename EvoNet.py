import collections
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Operation import *
from Operation import Operations_4_name, Operations_4
from Genotype2net import dagnode


class ReductionCell(nn.Module):
    def __init__(self):
        super(ReductionCell, self).__init__()

        self.maxpool5x5 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)  # Maxpool5*5
        self.maxpool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Maxpool2*2
        self.maxpool3x3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Maxpool3*3
        self.avgpool5x5 = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)  # Avgpool5*5
        self.identity = nn.Identity()

    def forward(self, input):
        out1 = self.maxpool5x5(input)
        out2 = self.maxpool2x2(input)
        out3 = self.maxpool3x3(input)
        out4 = self.avgpool5x5(input)
        output = out1 + out2 + out3 + out4
        return output


class AuxHeadCIFAR(nn.Module):
    def __init__(self, C_in, classes):
        super(AuxHeadCIFAR, self).__init__()

        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False)
        self.conv1 = nn.Conv2d(C_in, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 768, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(768)
        self.relu3 = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(768, classes)

    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn1.train()
            self.bn2.train()
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.global_pool(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class SearchSpace(nn.Module):
    def __init__(self, inner_genotype, input_channels=3, output_channels=64, num_classes=10, device=None, p=None,
                 stem_multiplier=1, init=False):
        super(SearchSpace, self).__init__()
        stack_genotype = inner_genotype[:3]
        rounded_stack_genotype = [round(x) for x in stack_genotype]
        net_genotype = inner_genotype[3:]
        self.device = device
        self.in_channels = input_channels
        self.out_channels = output_channels
        self.init = init

        stem = stem_multiplier
        channels = stem * self.out_channels
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False)
        )

        inverted_block = net_genotype[:16]
        self.invertedBlock = nn.Sequential()
        L = 0
        j = 0
        for i in range(inverted_block[0]):
            S = L + 1
            L += 3
            layer_type = inverted_block[S:L + 1]
            block = InvertedBottleneckBlock(layer_type, channels)
            self.invertedBlock.append(block)
            j += 1

        self.model1 = nn.ModuleList([copy.deepcopy(self.invertedBlock) for _ in range(rounded_stack_genotype[0] + 1)])

        self.conv = nn.Conv2d(channels, output_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(output_channels)

        self.reduction = ReductionCell()

        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        normal_block = net_genotype[16:36]
        normal_block[0] = 1
        L = 0
        j = 0
        self.normal_dag = collections.defaultdict()
        for i in range(5):
            S = L
            L += j + 2
            node_adj = normal_block[S:L]
            self.normal_dag[j] = dagnode(j, node_adj[:-1], node_adj[-1])
            j += 1


        self.normal_ops = NormalBlock(self.normal_dag, self.out_channels, self.device)

        self.model2 = nn.ModuleList([copy.deepcopy(self.normal_ops) for _ in range(rounded_stack_genotype[1] + 1)])

        model3_channels = self.normal_ops.out_channels

        residual_block = net_genotype[36:]
        self.residualBlock = nn.Sequential()
        L = 0
        j = 0

        for i in range(residual_block[0] + 1):
            S = L + 1
            L += 2
            layer_type = residual_block[S:L + 1]
            block = BottleneckBlock(layer_type, model3_channels)
            self.residualBlock.append(block)
            j += 1

        self.model3 = nn.ModuleList([copy.deepcopy(self.residualBlock) for _ in range(rounded_stack_genotype[2] + 1)])

        self.last_conv = nn.Sequential(
            nn.Conv2d(model3_channels, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=False)
        )

        # self.fc = nn.Linear(3, num_classes)
        # self.drop = nn.Dropout(p=p)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(model3_channels, num_classes)

        self.auxhead = AuxHeadCIFAR(model3_channels, num_classes)

        if self.init:
            self.init_parameters()

    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)

    def _skip_add(self, identity, out, name_prefix):
        """
        将 identity 调整到与 out 同形状后相加。
        在第一次出现通道不一致时，会创建一个 1×1 Conv 并注册为子模块，
        以后同样位置复用同一个投影层。
        """
        # 1) 空间尺寸对齐
        if identity.shape[2:] != out.shape[2:]:
            identity = F.interpolate(identity,
                                     size=out.shape[2:],
                                     mode='bilinear',
                                     align_corners=False)

        # 2) 通道数对齐
        if identity.shape[1] != out.shape[1]:
            proj_name = f"{name_prefix}_proj"
            if not hasattr(self, proj_name):  # 第一次才建
                proj_layer = nn.Conv2d(identity.size(1),
                                       out.size(1),
                                       kernel_size=1,
                                       bias=False).to(out.device)
                setattr(self, proj_name, proj_layer)
            identity = getattr(self, proj_name)(identity)

        return out + identity

    # residual connect
    def forward(self, x, return_aux=False):
        out = self.stem(x)
        for idx, block in enumerate(self.model1):
            out = block(out)
            identity = out
        out = self._skip_add(identity, out, name_prefix=f"m1_first")

        out = self.conv(out)
        out = self.bn(out)
        out = self.reduction(out)

        aux_logits = None

        for idx, block in enumerate(self.model2):
            out = block(out)
            identity = out
        out = self._skip_add(identity, out, name_prefix=f"m2_second")
        out = self.reduction(out)

        if return_aux:
            aux_logits = self.auxhead(out, bn_train=self.training)

        for idx, block in enumerate(self.model3):
            out = block(out)
            identity = out
        out = self._skip_add(identity, out, name_prefix=f"m3_third")

        out = self.global_pooling(out)
        out = self.fc(out.view(out.size(0), -1))

        if return_aux:
            return out, aux_logits
        return out

def invertedType2Truevalue(x):
    temp_genotype = copy.deepcopy(x)

    if temp_genotype[0] == 0:
        temp_genotype[0] = 1
    else:
        temp_genotype[0] = 2

    if temp_genotype[1] == 0:
        temp_genotype[1] = 2
    elif temp_genotype[1] == 1:
        temp_genotype[1] = 4
    elif temp_genotype[1] == 2:
        temp_genotype[1] = 8

    if temp_genotype[2] == 0:
        temp_genotype[2] = 3
    elif temp_genotype[2] == 1:
        temp_genotype[2] = 5
    elif temp_genotype[2] == 2:
        temp_genotype[2] = 7
    return temp_genotype


class InvertedBottleneckBlock(nn.Module):
    def __init__(self, inner_genotype, in_channels):
        super(InvertedBottleneckBlock, self).__init__()
        temp_genotype = invertedType2Truevalue(inner_genotype)
        self.stride = temp_genotype[0]

        # self.ops = nn.ModuleList()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * temp_genotype[1], kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(in_channels * temp_genotype[1]),
            nn.ReLU(inplace=False)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels * temp_genotype[1], in_channels * temp_genotype[1],
                      kernel_size=temp_genotype[2], stride=temp_genotype[0], padding=temp_genotype[2] // 2,
                      groups=in_channels * temp_genotype[1]),
            nn.BatchNorm2d(in_channels * temp_genotype[1]),
            nn.ReLU(inplace=False)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels * temp_genotype[1], in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.stride == 2:
            x_trans = F.interpolate(x, size=(out.size(2), out.size(3)), mode='bilinear', align_corners=False)
            out = out + x_trans
        if self.stride == 1:
            out = out + x
        out = F.interpolate(out, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return out


def residualType2Truevalue(x):
    temp_genotype = copy.deepcopy(x)

    if temp_genotype[0] == 0:
        temp_genotype[0] = 2
    elif temp_genotype[0] == 1:
        temp_genotype[0] = 4
    elif temp_genotype[0] == 2:
        temp_genotype[0] = 8

    if temp_genotype[1] == 0:
        temp_genotype[1] = 3
    elif temp_genotype[1] == 1:
        temp_genotype[1] = 5
    elif temp_genotype[1] == 2:
        temp_genotype[1] = 7
    return temp_genotype


class BottleneckBlock(nn.Module):
    def __init__(self, inner_genotype, in_channels):
        super(BottleneckBlock, self).__init__()
        temp_genotype = residualType2Truevalue(inner_genotype)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // temp_genotype[0], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels // temp_genotype[0]),
            nn.ReLU(inplace=False)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels // temp_genotype[0], in_channels // temp_genotype[0],
                      kernel_size=temp_genotype[1], stride=1, padding=temp_genotype[1] // 2),
            nn.BatchNorm2d(in_channels // temp_genotype[0]),
            nn.ReLU(inplace=False)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels // temp_genotype[0], in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        # print('x.shape', x.shape)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        x_trans = F.interpolate(x, size=(out.size(2), out.size(3)), mode='bilinear', align_corners=False)
        out = out + x_trans
        return out


class NormalBlock(nn.Module):
    def __init__(self, dag, channels, device,
                 drop_path_keep_prob=None, steps=0):
        super(NormalBlock, self).__init__()

        self.steps = steps
        self.drop_path_keep_prob = drop_path_keep_prob
        self.device = device
        self.dag = dag

        self.channels = channels

        self.num_node = len(dag)

        self.ops = nn.ModuleList()

        self.used = np.array([0] * (self.num_node + 1))

        for i in range(self.num_node):
            node = Node(self.dag[i], self.channels, self.device)
            self.ops.append(node)
            self.used[np.where(np.array(self.dag[i].adj_node) == 1)[0]] = 1

        self.concat = [i for i in range(self.num_node + 1) if self.used[i] == 0]
        self.out_channels = self.channels * len(self.concat) if self.concat else self.channels
        self.maybe_calibrate_size = MaybeCalibrateSize(self.channels, self.channels)

    def forward(self, input):
        input_transformed = self.maybe_calibrate_size(input)
        states = [input_transformed]

        for i in range(self.num_node):
            out = self.ops[i](states.copy())
            states.append(out)

        if not self.concat:
            print('The concat list is empty!!!!!!')

        if len(states) == 1:
            out = states[self.concat[0]] if self.concat else states[0]
        else:
            if self.concat:
                out = torch.cat([states[i] for i in self.concat], dim=1)
            else:
                out = states[-1]
        return out


class Node(nn.Module):
    def __init__(self, dag_node, channels, device, stride=1,
                 drop_path_keep_prob=None):
        super(Node, self).__init__()

        self.drop_path_keep_prob = drop_path_keep_prob
        self.device = device
        self.adj_node = dag_node.adj_node
        self.node_id = dag_node.node_id
        self.node_name = Operations_4_name[dag_node.op_id]
        self.Operation = Operations_4

        self.channels = channels
        self.stride = stride
        self.Factor_flag = any(self.adj_node[:])
        self.ops = self.Operation[dag_node.op_id](self.channels, self.channels, self.stride)

    def forward(self, x_input):
        if all(idx == 0 for idx in self.adj_node):
            shape = x_input[0].shape
            x = torch.zeros(shape, device=x_input[0].device)
        else:

            x = sum([x for i, x in enumerate(x_input) if self.adj_node[i] == 1])

        x = self.ops(x)

        return x
