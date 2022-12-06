"""Model library constants."""

from cyclops.utils.file import join

USE_CASES = {
    "mortality_decompensation": ["gemini", "mimiciv"],
    "delirium": ["gemini", "mimiciv"],
}
DATASETS = ["mimiciv", "gemini"]
DATA_TYPES = ["tabular", "temporal", "combined"]

TASKS = {"binary_classification": ["mortality_decompensation", "delirium"]}

CONFIG_FILE = join("configs", "models.yaml")
SAVE_DIR = join("/mnt", "cyclops", "checkpoints")
