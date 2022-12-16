"""NIH Chest X-ray dataset.

The dataset contains 112,120 frontal-view X-ray images of 30,805 unique patients with
the text-mined fourteen disease image labels.

"""
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import Subset
from torchvision import transforms
from torchxrayvision.datasets import NIH_Dataset, XRayCenterCrop, XRayResizer


class NIHCXRDataset:
    """NIH Chest X-ray dataset."""

    def __init__(self, cfg_path: str = "drift_detection/datasets/configs/nihcxr.yaml"):

        self.cfg = OmegaConf.load(cfg_path)

        self.image_path = self.cfg.image_path
        self.csv_path = self.cfg.csv_path
        self.image_size = self.cfg.image_size
        self.views = self.cfg.views
        self.unique_patients = self.cfg.unique_patients
        self.subset_size = self.cfg.subset_size

        self.dataset = NIH_Dataset(
            self.image_path,
            self.csv_path,
            views=list(self.views),
            unique_patients=self.unique_patients,
            transform=transforms.Compose(
                [XRayCenterCrop(), XRayResizer(224, engine="cv2")]
            ),
        )

        if self.subset_size is not None:
            indices = np.random.randint(0, len(self.dataset), size=self.subset_size)
            self.dataset = Subset(self.dataset, indices)

        self.metadata = self.dataset.dataset.csv.iloc[indices, :]
        self.metadata = pd.concat(
            [
                self.metadata,
                pd.DataFrame(
                    data=self.dataset.dataset.labels[indices, :],
                    index=indices,
                    columns=self.dataset.dataset.pathologies,
                ),
            ],
            axis=1,
        )

        self.metadata_mapping = self.cfg.metadata_mapping

    def get_data(self) -> Tuple[torch.utils.data.Dataset, pd.DataFrame, dict]:
        """Get the dataset and metadata.

        Returns
        -------
        data:
            dataset of scans
        metadata:
            metadata dataframe
        metadata_mapping:
            dictionary mapping columns
            names in metadata to standard names

        """
        return self.dataset, self.metadata, self.metadata_mapping
