import logging
import os
import copy
import argparse
plt = None

from EvoNet import *
from dataPrepare import build_train_Optimizer_Loss, build_cifar10_train_valid_test_loader, \
    build_cifar100_train_valid_test_loader


def count_parameters_in_MB(model):
    return sum(v.numel() for v in model.parameters()) / 1e6

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
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--cutmix', type=int, default=0)
    parser.add_argument('--cutout', type=int, default=1)
    args = parser.parse_args()
    cuda_index = args.cuda
    cutmix = args.cutmix

    use_cutout_flag = args.cutout == 1
    if use_cutout_flag:
        os.environ['USE_CUTOUT'] = '1'
    else:
        os.environ['USE_CUTOUT'] = '0'
    lr_list = []
    train_loss = []
    train_accuracy = []
    test_accuracy = []
    os.makedirs('results100', exist_ok=True)
    solution = [4, 4, 4,
                4, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2,
                1, 3, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3,
                4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    bs = int(os.environ.get('BATCH_SIZE', '256'))

    def get_device(default_cuda):
        if torch.cuda.is_available():
            index = cuda_index if cuda_index >= 0 else default_cuda
            return torch.device(f"cuda:{index}")
        else:
            return torch.device("cpu")

    device = torch.device(f"cuda:{cuda_index}")
    train_loader, test_loader = build_cifar100_train_valid_test_loader(bs)
    model = SearchSpace(solution, num_classes=100, device=device, p=0.05, stem_multiplier=2, init=True).to(device)
    epochs = 600
    print('!!!!!!!!!!Model Parameters: {} MB'.format(count_parameters_in_MB(model)))

    env_epochs = os.environ.get('EPOCHS')
    if env_epochs:
        epochs = int(env_epochs)

    train_criterion, eval_criterion, optimizer, scheduler = build_train_Optimizer_Loss(
        model, 0.9, 0.1, 5e-4, epochs, device=device, label_smoothing=0.1
    )


    aux_weight = 0.4

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

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
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
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
        print(f'Accuracy on the CIFAR-100 train images: {train_acc:.2f}%')
        print(f'Accuracy on the CIFAR-100 test images: {test_acc:.2f}%')
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

    file_name = f'results100/train_accuracy.txt'
    with open(file_name, 'w') as f:
        f.write(", ".join(map(str, train_accuracy)))

    file_name = f'results100/test_accuracy.txt'
    with open(file_name, 'w') as f:
        f.write(", ".join(map(str, test_accuracy)))
