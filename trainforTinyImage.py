import logging
import os
import copy
import argparse

plt = None
import torchvision.models as models

from EvoNet import *
from dataPrepare import build_train_Optimizer_Loss
from dataPrepare import build_train_tiny_imagenet_train_val_loader


def count_parameters_in_MB(model):
    return sum(v.numel() for v in model.parameters()) / 1e6


class ModelEMA(object):
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            if k in msd:
                v.copy_(v * self.decay + msd[k] * (1.0 - self.decay))

    def to(self, device):
        self.ema_model.to(device)
        return self


class ResNet18Wrapper(nn.Module):
    def __init__(self, num_classes=200):
        super(ResNet18Wrapper, self).__init__()
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.model = models.resnet18(weights=weights)
        except AttributeError:
            self.model = models.resnet18(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x, return_aux=False):
        out = self.model(x)
        if return_aux:
            return out, None
        return out


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


def visualize_conv_kernels(model):
    if plt is None:
        return
    for name, param in model.named_parameters():
        if len(param.size()) == 4:
            kernels = param.cpu().data
            num_kernels = kernels.shape[0]
            import matplotlib.pyplot as plt_local
            for i in range(min(num_kernels, 16)):
                kernel = kernels[i, 0, :, :]
                kernel_min = kernel.min()
                kernel_max = kernel.max()
                kernel = (kernel - kernel_min) / (kernel_max - kernel_min + 1e-5)
                plt_local.subplot(2, 8, i + 1)
                plt_local.imshow(kernel, cmap='gray')
                plt_local.axis('off')
            plt_local.suptitle(f'Visualization of Conv Kernels - {name}')
            plt_local.show()


def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    set_seed(2025)
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--cutmix', type=int, default=0)
    args = parser.parse_args()
    cuda_index = args.cuda
    cutmix = args.cutmix

    lr_list = []
    train_loss = []
    train_accuracy = []
    test_accuracy = []
    os.makedirs('results_tiny', exist_ok=True)
    log_path = f'results_tiny/log_{indice}.txt'

    solution = [4, 4, 4,
                4, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2,
                1, 3, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3,
                4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]


    bs = int(os.environ.get('BATCH_SIZE', '64'))
    tiny_root = os.environ.get('TINY_ROOT', './data/tiny-imagenet-200')

    device = torch.device(f"cuda:{cuda_index}")
    train_loader, test_loader = build_train_tiny_imagenet_train_val_loader(tiny_root, bs)
    model = SearchSpace(solution, num_classes=200, device=device, p=0.05, stem_multiplier=2, init=True).to(device)
    epochs = 250
    print('!!!!!!!!!!Model Parameters: {} MB'.format(count_parameters_in_MB(model)))

    lr = 0.025
    weight_decay = 5e-4

    train_criterion, eval_criterion, optimizer, _ = build_train_Optimizer_Loss(
        model, 0.9, lr, weight_decay, epochs, device=device, label_smoothing=0.1
    )

    warmup_epochs = max(5, int(0.05 * epochs))
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_epochs, epochs)

    cutmix_alpha = 1.0
    cutmix_prob = 1.0

    aux_weight = 0.4

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            use_cutmix = cutmix_alpha > 0.0 and np.random.rand() < cutmix_prob
            if use_cutmix:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, alpha=cutmix_alpha)
                outputs, aux_outputs = model(inputs, return_aux=True)
                loss_main = train_criterion(outputs, targets_a) * lam + train_criterion(outputs, targets_b) * (
                            1.0 - lam)
            else:
                outputs, aux_outputs = model(inputs, return_aux=True)
                loss_main = train_criterion(outputs, labels)

            loss_main_total = loss_main

            if aux_outputs is not None:
                if use_cutmix:
                    loss_aux = train_criterion(aux_outputs, targets_a) * lam + train_criterion(aux_outputs,
                                                                                               targets_b) * (1.0 - lam)
                else:
                    loss_aux = train_criterion(aux_outputs, labels)
                loss = loss_main_total + aux_weight * loss_aux
            else:
                loss = loss_main_total

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
        train_loss.append(epoch_loss)

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
        print(f'Accuracy on the Tiny-ImageNet train images: {train_acc:.2f}%')
        print(f'Accuracy on the Tiny-ImageNet val images: {test_acc:.2f}%')
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)
        with open(log_path, 'a') as lf:
            lf.write(f'Epoch {epoch + 1}: loss={epoch_loss:.4f}, train_acc={train_acc:.2f}, test_acc={test_acc:.2f}\n')

    file_name = f'results_tiny/train_accuracy.txt'
    with open(file_name, 'w') as f:
        f.write(", ".join(map(str, train_accuracy)))

    file_name = f'results_tiny/test_accuracy.txt'
    with open(file_name, 'w') as f:
        f.write(", ".join(map(str, test_accuracy)))
