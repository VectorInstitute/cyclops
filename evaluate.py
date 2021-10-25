import argparse
from torch.utils.data import DataLoader

from dataset import get_dataset, split_train_and_val
from model import get_model
import datapipeline.config as conf

from evidently.dashboard import Dashboard
from evidently.tabs import ClassificationPerformanceTab

def prepare_args():
    parser = argparse.ArgumentParser(description="ML OPS Testing")
    # model configs
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--model_path", type=str, default="./model.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)

    # data configs
    parser.add_argument("--dataset", type=str, default="gemini")
    parser.add_argument("--dataset_config", type=str, default="datapipeline/delirium.config")

    args = parser.parse_args()
    return args


def to_loader(dataset, args, shuffle=False):
    return DataLoader(dataset,
                      batch_size=args.batch_size,
                      shuffle=shuffle,
                      num_workers=args.num_workers,
                      pin_memory=True)


def predict(model, loader):
    output = []
    for (data, target) in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).to(data.dtype)

        out = model(data)
        output.append(out.squeze(dim=1))
    return output

def evaluate():
    args = prepare_args()
    config = conf.read_config(args.dataset_config)
    train_dataset, val_dataset, test_dataset = get_dataset(args.dataset)(args)
    val_loader = to_loader(val_dataset, args, shuffle=False)
    test_loader = to_loader(test_dataset, args, shuffle=False)

    # set data dimensions automatically based on dataset
    args.data_dim = train_dataset.dim()

    model = get_model(args.model)(args).to(device)
    
    # get prediction results for both val and test
    reference['prediction'] = predict(model, val_loader)
    production['prediction'] = predict(model, test_loader)

    column_mapping = {}

    column_mapping['target'] = config.target
    column_mapping['prediction'] = 'prediction'
    column_mapping['numerical_features'] = config.numerical_features
    column_mapping['categorical_features'] = config.categorical_features

    perfomance_dashboard = Dashboard(tabs=[ClassificationPerformanceTab])
    perfomance_dashboard.calculate(reference, production, column_mapping=column_mapping)

    perfomance_dashboard.save("../performance_report.html")  # TODO: filename should be a parameter

if __name__ == "__main__":
    evaluate()
    