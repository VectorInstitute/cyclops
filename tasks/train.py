import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tasks.dataset import get_dataset, split_train_and_val
from tasks.model import get_model
from tasks.utils.utils import AverageBinaryClassificationMetric

from mlflow import log_params

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_loader(dataset, args, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def validate(model, val_loader, loss_fn):

    metric = AverageBinaryClassificationMetric()
    for (data, target) in val_loader:

        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).to(data.dtype)

        output = model(data)

        loss = loss_fn(output.squeeze(dim=1), target)

        metric.add_step(loss, output, target)

    val_metric_dict = metric.compute_metrics()

    dict_str = " ".join(f"{k}: {v:.2f}" for k, v in val_metric_dict.items())

    to_print = f"validation: {dict_str}"

    # put this in a logger
    print(to_print)


def train(model, optimizer, dataloader, loss_fn, num_epochs):

    metric = AverageBinaryClassificationMetric()

    for e in tqdm(range(num_epochs)):

        for (data, target) in dataloader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).to(data.dtype)

            optimizer.zero_grad()

            output = model(data)

            loss = loss_fn(output.squeeze(), target)

            loss.backward()

            optimizer.step()

            metric.add_step(loss.detach(), output.detach(), target)

        epoch_metric_dict = metric.compute_metrics()

        dict_str = " ".join(f"{k}: {v:.2f}" for k, v in epoch_metric_dict.items())

        to_print = f"epoch {e} {dict_str}"

        # put this in a logger
        print(to_print)

        metric.reset()


def main(args):
    # MLflow parameters
    mlflow_params_dict = {
        "dataset": args.dataset,
        "no of workers": args.num_workers,
        "train data shuffle": args.shuffle,
        "model": args.model,
        "learning rate": args.lr,
        "no of epochs": args.num_epochs,
        "batch size": args.batch_size,
    }
    log_params(mlflow_params_dict)

    train_dataset, val_dataset = get_dataset(args.dataset)(args)

    train_loader = to_loader(train_dataset, args, args.shuffle)
    val_loader = to_loader(val_dataset, args, shuffle=False)

    # set data dimensions automatically based on dataset
    args.data_dim = train_dataset.dim()

    model = get_model(args.model)(args).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(model, optimizer, train_loader, loss_fn, num_epochs=args.num_epochs)

    torch.save(model.state_dict(), args.model_path)

    validate(model, val_loader, loss_fn)
