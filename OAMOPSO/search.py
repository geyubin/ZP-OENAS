import os
import random
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from Mopso import *
from dataPrepare import build_search_cifar10_train_valid_test_loader, \
    build_search_cifar100_train_valid_test_loader, \
    build_search_tiny_imagenet_train_val_loader, build_search_cinic10_train_valid_test_loader

curr_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.dirname(curr_dir)
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)


def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(2025)
    w = 0.729
    c1 = 1.46
    c2 = 1.46
    particals = 20
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--constraint', type=float, default=2.5)
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "cinic10", "tiny"])
    args = parser.parse_args()
    cuda_index = args.cuda
    constraint = args.constraint

    if torch.cuda.is_available():
        if cuda_index >= 0:
            device = torch.device(f"cuda:{cuda_index}")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    cycle_ = 25
    mesh_div = 10
    thresh = 20
    min_ = np.zeros(50)
    max_ = np.array([4, 4, 4,
                     4, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2,
                     1, 3, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3,
                     4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

    if args.dataset == "cifar10":
        train_loader, valid_loader = build_search_cifar10_train_valid_test_loader(128)
    if args.dataset == "cifar100":
        train_loader, valid_loader = build_search_cifar100_train_valid_test_loader(128)
    if args.dataset == "cinic10":
        train_loader, valid_loader = build_search_cinic10_train_valid_test_loader(128)
    if args.dataset == "tiny":
        tiny_root = os.environ.get('TINY_ROOT', os.path.join(proj_root, 'data', 'tiny-imagenet-200'))
        train_loader, valid_loader = build_search_tiny_imagenet_train_val_loader(tiny_root, 64)

    start_time = time.time()

    mopso_ = Mopso(particals, w, c1, c2, max_, min_, thresh, train_loader, valid_loader, mesh_div, device,
                   constraint,
                   use_elite_init=True, use_dual=True, use_elite_update=True)
    pareto_in, pareto_fitness = mopso_.done(cycle_)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(pareto_in)
    print(pareto_fitness)
    print(f"运行时间: {elapsed_time} 秒")

    file = f'search_results{args.dataset}/{args.dataset}_PF_position_{str(constraint).replace(".", "_")}.txt'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w') as f:
        f.write(", ".join(map(str, pareto_in)))

    file = f'search_results{args.dataset}/{args.dataset}_PF_fitness_{str(constraint).replace(".", "_")}.txt'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w') as f:
        f.write(", ".join(map(str, pareto_fitness)))


if __name__ == "__main__":
    main()
