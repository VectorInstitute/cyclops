import configargparse
import torch
import pandas as pd
import datetime
from torch.utils.data import DataLoader

from model import get_model
from dataset import pandas_to_dataset
from train import to_loader
import datapipeline.config as conf
from utils.utils import AverageBinaryClassificationMetric

import mlflow
from mlflow import log_params


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prepare_args(file = False):
    if not file:
        parser = configargparse.ArgumentParser()
    else:
        parser = configargparse.ArgumentParser(default_config_files=[file])

    parser.add('-c', '--config_file', is_config_file=True, help='config file path')

    # model configs
    parser.add("--model", type=str, default="mlp")
    parser.add("--model_path", type=str, default="./model.pt")
    parser.add("--batch_size", type=int, default=64)
    parser.add("--num_workers", type=int, default=0)
    parser.add("--threshold", type=float, default=0.5)

    # data configs
    parser.add("--input", type=str, default = "../test.csv")
    parser.add("--output", type=str, default = "../result.csv")

    # used by gemini data pipeline
    parser.add("--dataset_config", type=str, default="config/gemini_data.cfg")

    args, unknown = parser.parse_known_args()
    return args

def predict(model, loader):
    output = []
    metric = AverageBinaryClassificationMetric()
    for (data, target) in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).to(data.dtype)

        out = model(data)
        metric.add_step(0, out, target)
        output = output + out.squeeze(dim=1).tolist()

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
    
