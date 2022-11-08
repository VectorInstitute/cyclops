"""GEMINI dataset.

GEMINI dataset for risk of mortality use-case.

"""
from omegaconf import OmegaConf

from torchxrayvision.datasets import NIH_Dataset
from torchvision import transforms
from torchxrayvision.datasets import XRayCenterCrop, XRayResizer 
from torch.utils.data import Subset
import numpy as np
import torch
import pandas as pd
from typing import Tuple
from gemini.query import get_gemini_data
from gemini.utils import get_label, import_dataset_hospital, normalize, process, scale
import random

class GEMINIDataset:
    """GEMINI dataset."""
    def __init__(self, cfg_path: str = 'drift_detection/datasets/configs/gemini.yaml'):

        self.cfg = OmegaConf.load(cfg_path)

        admin_data, x, y = get_gemini_data(self.cfg.path)
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test), _, admin_data = import_dataset_hospital(admin_data, 
                                                                                                          x, 
                                                                                                          y, 
                                                                                                          self.cfg.dataset, 
                                                                                                          self.cfg.metadata_mapping.hospital_type,
                                                                                                          self.cfg.metadata_mapping.encounter_id,
                                                                                                          self.cfg.seed,
                                                                                                          self.cfg.shuffle,
                                                                                                          self.cfg.train_frac)

        # Normalize, scale, and process training data
        X_train = normalize(admin_data, X_train, self.cfg.aggregation_type, self.cfg.timesteps)
        X_train = scale(X_train)
        X_train = process(X_train, self.cfg.aggregation_type, self.cfg.timesteps)
        
        # Normalize, scale, and process validation data
        X_valid = normalize(admin_data, X_valid, self.cfg.aggregation_type, self.cfg.timesteps)
        X_valid = scale(X_valid)
        X_valid = process(X_valid, self.cfg.aggregation_type, self.cfg.timesteps)

        # Normalize, scale, and process test data
        X_test = normalize(admin_data, X_test, self.cfg.aggregation_type, self.cfg.timesteps)
        X_test = scale(X_test)
        X_test = process(X_test, self.cfg.aggregation_type, self.cfg.timesteps)

        # Get labels for aggreation type
        if self.cfg.aggregation_type != "time":
            y_train = get_label(admin_data, X_train, self.cfg.outcome)
            y_valid = get_label(admin_data, X_valid, self.cfg.outcome)
            y_test = get_label(admin_data, X_test, self.cfg.outcome)

        self.features = {'X_train': X_train,
                         'y_train': y_train,
                         'X_valid': X_valid, 
                         'y_valid': y_valid,
                         'X_test': X_test,
                         'y_test': y_test
                          }
        
        self.metadata = admin_data
        self.metadata_mapping = self.cfg.metadata_mapping
    
    def get_data(self) -> Tuple[np.ndarray, pd.DataFrame, dict]:
        '''
        Returns:
                data: 
                    dictionary of features and labels:
                    "X_train": training data
                    "y_train": training labels
                    "X_valid": validation data
                    "y_valid": validation labels
                    "X_test": test data
                    "y_test": test labels
                metadata: metadata dataframe
                metadata_mapping: dictionary mapping columns names in metadata to standard names
                '''
        return self.features, self.metadata, self.metadata_mapping
