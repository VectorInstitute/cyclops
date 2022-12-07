"""GEMINI dataset.

GEMINI dataset for risk of mortality use-case.

"""
from typing import Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# from gemini.utils import get_label, import_dataset_hospital, normalize, process, scale


class GEMINIDataset:
    """GEMINI dataset."""

    def __init__(self, cfg_path: str = "cyclops/monitor/datasets/configs/gemini.yaml"):

        self.cfg = OmegaConf.load(cfg_path)

        self.features = None
        self.metadata = None
        self.metadata_mapping = self.cfg.metadata_mapping

        # admin_data, x, y = get_gemini_data(self.cfg.path)
        # admin_data[self.cfg.metadata_mapping["targets"]] = y

        # (x_train, y_train), (x_valid, y_valid), (x_test, y_test), _, admin_data =
        # import_dataset_hospital(admin_data,
        # x,
        # y,
        # self.cfg.dataset,
        # self.cfg.metadata_mapping.hospital_type,
        # self.cfg.metadata_mapping.encounter_id,
        # self.cfg.seed,
        # self.cfg.shuffle,
        # self.cfg.train_frac)

        # # Normalize, scale, and process training data
        # x_train = normalize(admin_data, x_train,
        #                     self.cfg.aggregation_type,
        #                     self.cfg.timesteps)
        # x_train = scale(x_train)
        # x_train = process(x_train, self.cfg.aggregation_type,
        #                   self.cfg.timesteps)

        # # Normalize, scale, and process validation data
        # x_valid = normalize(admin_data, x_valid,
        #                     self.cfg.aggregation_type,
        #                     self.cfg.timesteps)
        # x_valid = scale(x_valid)
        # x_valid = process(x_valid,
        #                   self.cfg.aggregation_type,
        #                   self.cfg.timesteps)

        # # Normalize, scale, and process test data
        # x_test = normalize(admin_data, x_test,
        #                    self.cfg.aggregation_type,
        #                    self.cfg.timesteps)
        # x_test = scale(x_test)
        # x_test = process(x_test,
        #                  self.cfg.aggregation_type,
        #                  self.cfg.timesteps)

        # # Get labels for aggreation type
        # if self.cfg.aggregation_type != "time":
        #     y_train = get_label(admin_data, x_train, self.cfg.outcome)
        #     y_valid = get_label(admin_data, x_valid, self.cfg.outcome)
        #     y_test = get_label(admin_data, x_test, self.cfg.outcome)

        # self.features = {'x': x,
        #                  'y': y,
        #      'x_train': x_train,
        #      'y_train': y_train,
        #      'x_valid': x_valid,
        #      'y_valid': y_valid,
        #      'x_test': x_test,
        #      'y_test': y_test
        #                   }

        # self.features = x
        # self.metadata = admin_data
        # self.metadata_mapping = self.cfg.metadata_mapping

    def get_data(self) -> Tuple[np.ndarray, pd.DataFrame, dict]:
        """Get data.

        Returns
        -------
        data:
            numpy array of features
        metadata:
            metadata dataframe
        metadata_mapping:
            dictionary mapping columns names
            in metadata to standard names

        """
        return self.features, self.metadata, self.metadata_mapping
