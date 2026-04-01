import logging
import os
import copy
import argparse

from EvoNet import *
from dataPrepare import build_train_Optimizer_Loss, build_cinic10_train_valid_test_loader


def count_parameters_in_MB(model):
    return sum(v.numel() for v in model.parameters()) / 1e6


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
    parser.add_argument('--cutout', type=int, default=1)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed is None:
        seed = np.random.randint(0, 100000)
    else:
        seed = args.seed
    print(f"Random Seed: {seed}")
    set_seed(seed)

    indice = args.indice
    cuda_index = args.cuda
    use_cutout_flag = args.cutout == 1
    if use_cutout_flag:
        os.environ['USE_CUTOUT'] = '1'
    else:
        os.environ['USE_CUTOUT'] = '0'

    lr_list = []
    train_loss = []
    train_accuracy = []
    test_accuracy = []
    os.makedirs('results_cinic10New', exist_ok=True)
    solution = [4, 4, 4,
                4, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2,
                1, 3, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3,
                4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    bs = int(os.environ.get('BATCH_SIZE', '128'))
    cinic_root = os.environ.get('CINIC10_ROOT', './data/cinic10')

    device = torch.device(f"cuda:{cuda_index}")
    train_loader, valid_loader, test_loader = build_cinic10_train_valid_test_loader(bs, root=cinic_root)
    model = SearchSpace(solution, num_classes=10, device=device, p=0.05, stem_multiplier=2, init=False).to(
        device)
    epochs = 600
    print('!!!!!!!!!!Model Parameters: {} MB'.format(count_parameters_in_MB(model)))

    env_epochs = os.environ.get('EPOCHS')
    if env_epochs:
        epochs = int(env_epochs)

    train_criterion, eval_criterion, optimizer, scheduler = build_train_Optimizer_Loss(
        model, 0.9, 0.025, 5e-4, epochs, device=device, label_smoothing=0.1
    )

    warmup_epochs = min(10, int(0.05 * epochs))
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_epochs, epochs)
    aux_weight = 0.4

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()


            if indice == 5 and np.random.random() > 0.5:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, alpha=1.0)
                outputs, aux_outputs = model(inputs, return_aux=True)
                loss_main = lam * train_criterion(outputs, targets_a) + (1 - lam) * train_criterion(outputs, targets_b)
                if aux_outputs is not None:
                    loss_aux = lam * train_criterion(aux_outputs, targets_a) + (1 - lam) * train_criterion(aux_outputs, targets_b)
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

            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_acc = 100 * correct_train / total_train
        test_acc = 100 * correct / total
        print(f'Accuracy on the CINIC-10 train images: {train_acc:.2f}%')
        print(f'Accuracy on the CINIC-10 test images: {test_acc:.2f}%')
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

    file_name = f'results_cinic10New/train_accuracy.txt'
    with open(file_name, 'w') as f:
        f.write(", ".join(map(str, train_accuracy)))

    file_name = f'results_cinic10New/test_accuracy.txt'
    with open(file_name, 'w') as f:
        f.write(", ".join(map(str, test_accuracy)))
