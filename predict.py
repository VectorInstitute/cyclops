import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader

from model import get_model
from dataset import pandas_to_dataset
from main import to_loader
import datapipeline.config as conf

from evidently.dashboard import Dashboard
from evidently.tabs import ClassificationPerformanceTab


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prepare_args():
    parser = argparse.ArgumentParser(description="ML OPS Testing")
    # model configs
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--model_path", type=str, default="./model.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)

    # data configs
    parser.add_argument("--input", type=str, default = "../test.csv")
    parser.add_argument("--output", type=str, default = "../result.csv")

    # used by gemini data pipeline
    parser.add_argument("--dataset_config", type=str, default="datapipeline/delirium.config")

    args = parser.parse_args()
    return args

def predict(model, loader):
    output = []
    for (data, target) in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).to(data.dtype)

        out = model(data)
        output.append(out.squeeze(dim=1))
    return output

def main(args):
    # read model
    model = get_model(args.model)(args).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # read data
    config = conf.read_config(args.dataset_config)
    data = pd.read_csv(input)
    dataset = pandas_to_dataset(data, config.feature_cols, config.target_cols)
    loader = to_loader(dataset, args)

    result = predict(model, loader)

    # save results csv
    data['prediction'] = result
    data.to_cvs(args.output)

if __name__ == "__main__":
    args = prepare_args()
    main(args)
    
