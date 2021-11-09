import argparse
import torch
import pandas as pd
import datetime
from torch.utils.data import DataLoader

from model import get_model
from dataset import pandas_to_dataset
from main import to_loader
import datapipeline.config as conf
from utils import AverageBinaryClassificationMetric

from mlflow import log_params


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prepare_args():
    parser = argparse.ArgumentParser(description="ML OPS Testing")
    # model configs
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--model_path", type=str, default="./model.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)

    # data configs
    parser.add_argument("--input", type=str, default = "../test.csv")
    parser.add_argument("--output", type=str, default = "../result.csv")

    # used by gemini data pipeline
    parser.add_argument("--dataset_config", type=str, default="datapipeline/delirium.config")

    args = parser.parse_args()
    return args

def predict(model, loader):
    output = []
    metric = AverageBinaryClassificationMetric()
    for (data, target) in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).to(data.dtype)

        out = model(data)
        output = output + out.squeeze(dim=1).tolist()
        metric.add_step(0, output, target)

    val_metric_dict = metric.compute_metrics()
    return output

def main(args):
    # read data
    exp = mlflow.get_experiment_by_name('Prediction')
    with mlflow.start_run(experiment_id=exp.experiment_id):
        config = conf.read_config(args.dataset_config)
        mlflow.log_dict(vars(config), 'dataset_config.json')
        mlflow.log_dict(vars(args), 'args.json')
        mlflow.log_params({'timestamp': datetime.datetime.now()})
        data = pd.read_csv(args.input)
        dataset = pandas_to_dataset(data, config.features, config.target)
        args.data_dim = dataset.dim()
        loader = to_loader(dataset, args)

        # read model
        model = get_model(args.model)(args).to(device)
        model.load_state_dict(torch.load(args.model_path))
        model.eval()

        result = predict(model, loader)

        # save results csv
        data['prediction'] = result
        data.loc[data['prediction'] >= args.threshold,'prediction'] = 1
        data.loc[data['prediction'] < args.threshold,'prediction'] = 0
        data.to_csv(args.output)
        mlflow.log_artifact(args.output)

if __name__ == "__main__":
    args = prepare_args()
    main(args)
    
