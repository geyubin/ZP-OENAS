import logging
import os
import argparse
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps

from EvoNet import *
from dataPrepare import build_train_Optimizer_Loss, build_cifar10_train_valid_test_loader, \
    build_cifar100_train_valid_test_loader


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(inputs, targets, alpha=1.0):
    if alpha <= 0.0:
        return inputs, targets, targets, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size).to(inputs.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
    inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size(2) * inputs.size(3)))
    targets_a = targets
    targets_b = targets[index]
    return inputs, targets_a, targets_b, lam


def build_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        if total_epochs <= warmup_epochs:
            return 1.0
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def count_parameters_in_MB(model):
    # for param in model.parameters():
    #     print(param.dtype)
    return sum(v.numel() for v in model.parameters()) / 1e6
    # return sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e6


def visualize_conv_kernels(model):
    for name, param in model.named_parameters():
        if len(param.size()) == 4:  # Conv2d权重：[out_channels, in_channels, H, W]
            kernels = param.cpu().data
            num_kernels = kernels.shape[0]
            plt.figure(figsize=(20, 5))
            for i in range(min(num_kernels, 16)):  # 最多画16个
                kernel = kernels[i, 0, :, :]
                # 做归一化处理
                kernel_min = kernel.min()
                kernel_max = kernel.max()
                kernel = (kernel - kernel_min) / (kernel_max - kernel_min + 1e-5)
                plt.subplot(2, 8, i + 1)
                plt.imshow(kernels[i, 0, :, :], cmap='gray')  # 取第0通道展示
                plt.axis('off')
            plt.suptitle(f'Visualization of Conv Kernels - {name}')
            plt.show()


def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=-1)
    args = parser.parse_args()
    cuda_index = args.cuda

    solution = [4, 4, 4,
                     4, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2,
                     1, 3, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3,
                     4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    lr_list = []
    train_loss = []
    train_accuracy = []
    test_accuracy = []
    device = torch.device(f"cuda:{cuda_index}")
    train_loader, test_loader = build_cifar10_train_valid_test_loader(128)
    model = SearchSpace(solution, num_classes=10, device=device, p=0.05, stem_multiplier=2, init=False).to(device)
    epochs = 600
    print('Model Parameters: {} MB'.format(count_parameters_in_MB(model)))

    train_criterion, eval_criterion, optimizer, scheduler = build_train_Optimizer_Loss(model, 0.9, 0.025, 5e-4,
                                                                                       epochs, device=device, label_smoothing=0.1)

    warmup_epochs = max(5, int(0.05 * epochs))
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_epochs, epochs)

    cutmix_alpha = 1.0
    cutmix_prob = 1.0

    aux_weight = 0.4

    print("\n" + "="*80)
    print(f"时间戳: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    print(f"参数:")
    print(f"  - 数据集 (Dataset):         CIFAR-10")
    print(f"  - 总轮数 (Epochs):            {epochs}")
    print(f"  - 批量大小 (Batch Size):      {train_loader.batch_size}")
    print(f"  - 初始学习率 (Initial LR):    {optimizer.param_groups[0]['lr']}")
    print(f"  - 辅助损失权重 (Aux Weight): {aux_weight}")
    print("="*80 + "\n")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            use_cutmix = cutmix_alpha > 0.0 and np.random.rand() < cutmix_prob
            if use_cutmix:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, alpha=cutmix_alpha)
                outputs, aux_outputs = model(inputs, return_aux=True)
                loss_main = train_criterion(outputs, targets_a) * lam + train_criterion(outputs, targets_b) * (1.0 - lam)
                if aux_outputs is not None:
                    loss_aux = train_criterion(aux_outputs, targets_a) * lam + train_criterion(aux_outputs, targets_b) * (1.0 - lam)
                    loss = loss_main + aux_weight * loss_aux
                else:
                    loss = loss_main
            else:
                outputs, aux_outputs = model(inputs, return_aux=True)
                loss_main = train_criterion(outputs, labels)
                if aux_outputs is not None:
                    loss_aux = train_criterion(aux_outputs, labels)
                    loss = loss_main + aux_weight * loss_aux
                else:
                    loss = loss_main

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_loss.append(epoch_loss)
        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        total_train = 0
        correct_train = 0
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_acc = 100 * correct_train / total_train
        test_acc = 100 * correct / total
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

        end_time = time.time()
        epoch_duration = end_time - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Loss: {epoch_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | "
              f"LR: {current_lr:.6f} | "
              f"耗时: {epoch_duration:.2f}s | "
              f"时间戳: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    file_name = f'results/train_accuracy.txt'
    with open(file_name, 'w') as f:
        f.write(", ".join(map(str, train_accuracy)))

    file_name = f'results/test_accuracy.txt'
    with open(file_name, 'w') as f:
        f.write(", ".join(map(str, test_accuracy)))
