"""Model library constants."""

import os

from cyclops.utils.file import join

USE_CASES = {
    "mortality_decompensation": ["gemini", "mimiciv"],
    "delirium": ["gemini", "mimiciv"],
}
DATASETS = ["mimiciv", "gemini"]
DATA_TYPES = ["tabular", "temporal", "combined"]

TASKS = {"binary_classification": ["mortality_decompensation", "delirium"]}

CONFIG_ROOT = join(os.path.dirname(__file__), "configs")

DATA_DIR = join("/mnt/data", "cyclops", "use_cases")
SAVE_DIR = join("/mnt/exp", "cyclops", "checkpoints")
