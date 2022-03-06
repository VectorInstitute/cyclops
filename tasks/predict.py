"""Prediction script."""

import logging
import datetime

import torch
import pandas as pd
import mlflow

from codebase_ops import get_log_file_path
from config import config_to_dict

from models.catalog import get_model
from models.metrics import AverageBinaryClassificationMetric
from models.dataset import pandas_to_dataset

from tasks.train import to_loader

from cyclops.utils.log import setup_logging


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict(model, loader):
    """Predict."""
    output = []
    metric = AverageBinaryClassificationMetric()
    for (data, target) in loader:
        data = data.to(DEVICE, non_blocking=True)
        target = target.to(DEVICE, non_blocking=True).to(data.dtype)

        out = model(data)
        metric.add_step(0, out, target)
        output = output + out.squeeze(dim=1).tolist()

    return output


def main(args):
    """Run prediction."""
    exp_name = "Prediction"
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        mlflow.create_experiment(exp_name)
        exp = mlflow.get_experiment_by_name(exp_name)
    with mlflow.start_run(experiment_id=exp.experiment_id):
        mlflow.log_dict(config_to_dict(args), "args.json")
        mlflow.log_params({"timestamp": datetime.datetime.now()})
        data = pd.read_csv(args.input)
        dataset = pandas_to_dataset(data, args.features, args.target, config=args)

        args.data_dim = dataset.dim()
        loader = to_loader(dataset, args)

        # read model
        model = get_model(args.model)(2, args.data_dim, [16, 8], 1, "silu").to(DEVICE)
        model.load_state_dict(torch.load(args.model_path))
        model.eval()

        _ = predict(model, loader)
