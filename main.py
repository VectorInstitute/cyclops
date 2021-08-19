import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_dataset, split_train_and_val
from model import get_model


def prepare_args():
    parser = argparse.ArgumentParser(description="ML OPS Testing")

    # model configs
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--hidden_dim", type=int, default=100)

    # data configs
    parser.add_argument("--dataset", type=str, default="fakedata")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")

    # used mostly for fake data, can take it out
    parser.add_argument("--data_dim", type=int, default=24)
    parser.add_argument("--data_len", type=int, default=10000)

    # training configs
    parser.add_argument('--lr', type=float, default=3e-4)

    args = parser.parse_args()
    return args


def to_loader(dataset, args, shuffle=False):
    return DataLoader(dataset,
                      batch_size=args.batch_size,
                      shuffle=shuffle,
                      num_workers=args.num_workers,
                      pin_memory=True)


def main(args):
    dataset = get_dataset(args.dataset)(args)
    train_dataset, val_dataset = split_train_and_val(dataset)

    train_loader = to_loader(train_dataset, args, args.shuffle)
    val_loader = to_loader(val_dataset, args, shuffle=False)

    # rest is TODO


if __name__ == "__main__":
    args = prepare_args()
    main(args)
