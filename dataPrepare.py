import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import random_split
from autoaugment import CIFAR10Policy, ImageNetPolicy
import torchvision.datasets as datasets

class Cutout(object):
    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def build_search_tiny_imagenet_train_val_loader(root, batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        normalize,
    ])

    num_workers = int(os.environ.get('NUM_WORKERS', 16))
    train_dir = os.path.join(root, 'train')
    val_dir = os.path.join(root, 'val')

    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    val_set = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def build_train_tiny_imagenet_train_val_loader(root, batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    num_workers = int(os.environ.get('NUM_WORKERS', 16))
    train_dir = os.path.join(root, 'train')
    test_dir = os.path.join(root, 'test')

    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    test_set = datasets.ImageFolder(test_dir, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def data_transform_cifar10():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, fill=128),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
        Cutout(16)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
    ])

    return train_transform, test_transform


def data_search_transform_cifar10():

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, fill=128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
    ])

    # 验证集图像处理
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
    ])

    return train_transform, test_transform


def data_transform_cinic10():
    cinic_mean = (0.47889522, 0.47227842, 0.43047404)
    cinic_std = (0.24205776, 0.23828046, 0.25874835)

    use_cutout = os.environ.get('USE_CUTOUT', '0') == '1'
    train_transforms = [
        transforms.RandomCrop(32, padding=4, fill=128),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize(cinic_mean, cinic_std),
    ]
    if use_cutout:
        train_transforms.append(Cutout(16))
    train_transform = transforms.Compose(train_transforms)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cinic_mean, cinic_std),
    ])

    return train_transform, test_transform


def data_search_transform_cinic10():

    cinic_mean = (0.47889522, 0.47227842, 0.43047366)
    cinic_std = (0.24205776, 0.23828046, 0.2592109)
    use_cutout = os.environ.get('USE_CUTOUT', '0') == '1'
    train_transforms = [
        transforms.RandomCrop(32, padding=4, fill=128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cinic_mean, cinic_std),
    ]
    if use_cutout:
        train_transforms.append(Cutout(16))
    train_transform = transforms.Compose(train_transforms)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cinic_mean, cinic_std),
    ])

    return train_transform, test_transform


def build_cifar10_train_valid_test_loader(batch_size):
    train_transform, valid_transform = data_transform_cifar10()
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    num_workers = int(os.environ.get('NUM_WORKERS', 16))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=valid_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


def build_cinic10_train_valid_test_loader(batch_size, root='./data/cinic10'):
    train_transform, valid_transform = data_transform_cinic10()
    num_workers = int(os.environ.get('NUM_WORKERS', 16))
    train_path = os.path.join(root, 'train')
    valid_path = os.path.join(root, 'valid')
    test_path = os.path.join(root, 'test')

    train_set = torchvision.datasets.ImageFolder(train_path, transform=train_transform)
    valid_set = torchvision.datasets.ImageFolder(valid_path, transform=valid_transform)
    test_set = torchvision.datasets.ImageFolder(test_path, transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader, test_loader


def build_search_cinic10_train_valid_test_loader(batch_size, root='./data/cinic10'):
    train_transform, valid_transform = data_search_transform_cinic10()
    num_workers = int(os.environ.get('NUM_WORKERS', 0))
    train_path = os.path.join(root, 'train')
    valid_path = os.path.join(root, 'valid')
    test_path = os.path.join(root, 'test')

    train_set = torchvision.datasets.ImageFolder(train_path, transform=train_transform)
    valid_set = torchvision.datasets.ImageFolder(valid_path, transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader


def build_search_cifar10_train_valid_test_loader(batch_size):
    train_transform, valid_transform = data_search_transform_cifar10()
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = random_split(train_set, [train_size, val_size])

    num_workers = int(os.environ.get('NUM_WORKERS', 16))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return train_loader, valid_loader


def data_transform_cifar100():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    ])

    # 验证集图像处理
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    ])

    return train_transform, test_transform


def data_train_transform_cifar100():
    train_transform = transforms.Compose([
        transforms.RandomCrop((32, 32), padding=4, fill=128),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        Cutout(16)
    ])

    # 验证集图像处理
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    ])

    return train_transform, test_transform


def build_search_cifar100_train_valid_test_loader(batch_size):
    train_transform, valid_transform = data_transform_cifar100()
    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = random_split(train_set, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return train_loader, valid_loader


def build_cifar100_train_valid_test_loader(batch_size):
    train_transform, valid_transform = data_train_transform_cifar100()
    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)

    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=valid_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return train_loader, test_loader


def build_train_Optimizer_Loss(model, momentum=0.9, lr_max=0.025, l2_reg=5e-4, epochs=50,
                               device=None, label_smoothing=0.0):
    device1 = device
    # model.to(device1)
    try:
        train_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device1)
        eval_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device1)
    except TypeError:
        train_criterion = nn.CrossEntropyLoss().to(device1)
        eval_criterion = nn.CrossEntropyLoss().to(device1)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr_max,
        momentum=momentum,
        weight_decay=l2_reg,
        nesterov=True
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    return train_criterion, eval_criterion, optimizer, scheduler
