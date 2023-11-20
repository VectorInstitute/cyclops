"""Test inputs."""

import random
from collections import namedtuple

import numpy as np
import pytest
import scipy as sp
from numpy.typing import ArrayLike

from .conftest import BATCH_SIZE, NUM_BATCHES, NUM_CLASSES, NUM_LABELS


Input = namedtuple("Input", ["target", "preds"])


def set_random_seed(seed: int) -> None:
    """Set random seed."""
    np.random.seed(seed)
    random.seed(seed)


def _inv_sigmoid(arr: ArrayLike) -> np.ndarray:
    """Inverse sigmoid function."""
    arr = np.asanyarray(arr)
    return np.log(arr / (1 - arr))


set_random_seed(42)

# binary cases
_binary_cases = (
    pytest.param(
        Input(
            target=np.random.randint(0, 2, (NUM_BATCHES, BATCH_SIZE)),
            preds=np.random.randint(0, 2, (NUM_BATCHES, BATCH_SIZE)),
        ),
        id="input[labels]",
    ),
    pytest.param(
        Input(
            target=np.random.randint(0, 2, (NUM_BATCHES, 1)),
            preds=np.random.randint(0, 2, (NUM_BATCHES, 1)),
        ),
        id="input[single-sample-labels]",
    ),
    pytest.param(
        Input(
            target=np.random.randint(0, 2, (NUM_BATCHES, BATCH_SIZE)),
            preds=np.random.rand(NUM_BATCHES, BATCH_SIZE),
        ),
        id="input[probs]",
    ),
    pytest.param(
        Input(
            target=np.random.randint(0, 2, (NUM_BATCHES, BATCH_SIZE)),
            preds=_inv_sigmoid(np.random.rand(NUM_BATCHES, BATCH_SIZE)),
        ),
        id="input[logits]",
    ),
)

# multiclass cases
_multiclass_cases = (
    pytest.param(
        Input(
            target=np.random.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE)),
            preds=np.random.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE)),
        ),
        id="input[labels]",
    ),
    pytest.param(
        Input(
            target=np.random.randint(0, NUM_CLASSES, (NUM_BATCHES, 1)),
            preds=np.random.randint(0, NUM_CLASSES, (NUM_BATCHES, 1)),
        ),
        id="input[single-sample-labels]",
    ),
    pytest.param(
        Input(
            target=np.random.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE)),
            preds=sp.special.softmax(
                np.random.randn(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES),
                axis=-1,
            ),
        ),
        id="input[probs]",
    ),
    pytest.param(
        Input(
            target=np.random.randint(0, NUM_CLASSES, (NUM_BATCHES, BATCH_SIZE)),
            preds=sp.special.log_softmax(
                (np.random.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)),
                axis=-1,
            ),
        ),
        id="input[logits]",
    ),
)

# multilabel cases
_multilabel_cases = (
    pytest.param(
        Input(
            target=np.random.randint(0, 2, (NUM_BATCHES, BATCH_SIZE, NUM_LABELS)),
            preds=np.random.randint(0, 2, (NUM_BATCHES, BATCH_SIZE, NUM_LABELS)),
        ),
        id="input[labels]",
    ),
    pytest.param(
        Input(
            target=np.random.randint(0, 2, (NUM_BATCHES, 1, NUM_LABELS)),
            preds=np.random.randint(0, 2, (NUM_BATCHES, 1, NUM_LABELS)),
        ),
        id="input[single-sample-labels]",
    ),
    pytest.param(
        Input(
            target=np.random.randint(0, 2, (NUM_BATCHES, BATCH_SIZE, NUM_LABELS)),
            preds=np.random.rand(NUM_BATCHES, BATCH_SIZE, NUM_LABELS),
        ),
        id="input[probs]",
    ),
    pytest.param(
        Input(
            target=np.random.randint(0, 2, (NUM_BATCHES, BATCH_SIZE, NUM_LABELS)),
            preds=_inv_sigmoid(np.random.rand(NUM_BATCHES, BATCH_SIZE, NUM_LABELS)),
        ),
        id="input[logits]",
    ),
)
