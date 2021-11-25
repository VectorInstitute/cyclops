import configargparse
import torch
import pandas as pd
import datetime
from torch.utils.data import DataLoader

from tasks.model import get_model
from tasks.dataset import pandas_to_dataset
from tasks.train import to_loader
from tasks.utils.utils import AverageBinaryClassificationMetric

import mlflow
from mlflow import log_params

import config.config as config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        data.to_csv(args.result_output)
        #mlflow.log_artifact(args.output)

    
