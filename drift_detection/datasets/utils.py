"""Utilities for loading datasets."""
from drift_detection.datasets import NIHCXRDataset


def load_dataset(dataset: str, cfg_path: str):
    
    datasets = {
        'nihcxr': NIHCXRDataset
    }
    
    return datasets[dataset](cfg_path).get_data()

