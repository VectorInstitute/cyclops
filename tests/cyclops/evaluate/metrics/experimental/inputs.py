"""Input data for tests of metrics in cyclops/evaluate/metrics/experimental."""
import random
from collections import namedtuple
from typing import Any

import array_api_compat as apc
import numpy as np
import pytest
import torch
from scipy.special import log_softmax

from cyclops.evaluate.metrics.experimental.utils.types import Array

from ..conftest import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES, NUM_CLASSES, NUM_LABELS


InputSpec = namedtuple("InputSpec", ["target", "preds"])


def set_random_seed(seed: int) -> None:
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _inv_sigmoid(arr: Array) -> Array:
    """Inverse sigmoid function."""
    xp = apc.array_namespace(arr)
    return xp.log(arr / (1 - arr))


set_random_seed(1)

# binary
# NOTE: the test will loop over the first dimension of the input
_binary_labels_0d = np.random.randint(0, 2, size=(NUM_BATCHES, 1))
_binary_preds_0d = np.random.randint(0, 2, size=(NUM_BATCHES, 1))
_binary_probs_0d = np.random.rand(NUM_BATCHES, 1)
_binary_labels_1d = np.random.randint(0, 2, size=(NUM_BATCHES, BATCH_SIZE))
_binary_preds_1d = np.random.randint(0, 2, size=(NUM_BATCHES, BATCH_SIZE))
_binary_probs_1d = np.random.rand(NUM_BATCHES, BATCH_SIZE)
_binary_labels_multidim = np.random.randint(
    0,
    2,
    size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
)
_binary_preds_multidim = np.random.randint(
    0,
    2,
    size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
)
_binary_probs_multidim = np.random.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)


def _binary_cases(*, xp: Any):
    """Return binary input cases for the given array namespace."""
    return (
        pytest.param(
            InputSpec(
                target=xp.asarray(_binary_labels_0d),
                preds=xp.asarray(_binary_preds_0d),
            ),
            id="input[single-element-labels]",
        ),
        pytest.param(
            InputSpec(
                target=xp.asarray(_binary_labels_1d),
                preds=xp.asarray(_binary_preds_1d),
            ),
            id="input[1d-labels]",
        ),
        pytest.param(
            InputSpec(
                target=xp.asarray(_binary_labels_multidim),
                preds=xp.asarray(_binary_preds_multidim),
            ),
            id="input[multidim-labels]",
        ),
        pytest.param(
            InputSpec(
                target=xp.asarray(_binary_labels_0d),
                preds=xp.asarray(_binary_probs_0d),
            ),
            id="input[single-element-probs]",
        ),
        pytest.param(
            InputSpec(
                target=xp.asarray(_binary_labels_0d),
                preds=xp.asarray(_inv_sigmoid(_binary_probs_0d)),
            ),
            id="input[single-element-logits]",
        ),
        pytest.param(
            InputSpec(
                target=xp.asarray(_binary_labels_1d),
                preds=xp.asarray(_binary_probs_1d),
            ),
            id="input[1d-probs]",
        ),
        pytest.param(
            InputSpec(
                target=xp.asarray(_binary_labels_1d),
                preds=xp.asarray(_inv_sigmoid(_binary_probs_1d)),
            ),
            id="input[1d-logits]",
        ),
        pytest.param(
            InputSpec(
                target=xp.asarray(_binary_labels_multidim),
                preds=xp.asarray(_binary_probs_multidim),
            ),
            id="input[multidim-probs]",
        ),
        pytest.param(
            InputSpec(
                target=xp.asarray(_binary_labels_multidim),
                preds=xp.asarray(_inv_sigmoid(_binary_probs_multidim)),
            ),
            id="input[multidim-logits]",
        ),
    )


def _multiclass_with_missing_class(
    *shape: Any,
    num_classes: int = NUM_CLASSES,
    xp: Any,
) -> Array:
    """Generate multiclass input where a class is missing.

    Args:
        shape: shape of the tensor
        num_classes: number of classes

    Returns
    -------
        tensor with missing classes

    """
    x = np.random.randint(0, num_classes, shape)
    x[x == 0] = 2
    return xp.asarray(x)


# multiclass
_multiclass_labels_0d = np.random.randint(0, NUM_CLASSES, size=(NUM_BATCHES, 1))
_multiclass_preds_0d = np.random.randint(0, NUM_CLASSES, size=(NUM_BATCHES, 1))
_multiclass_probs_0d = np.random.rand(NUM_BATCHES, 1, NUM_CLASSES)
_multiclass_labels_1d = np.random.randint(
    0,
    NUM_CLASSES,
    size=(NUM_BATCHES, BATCH_SIZE),
)
_multiclass_preds_1d = np.random.randint(0, NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE))
_multiclass_probs_1d = np.random.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)
_multiclass_labels_multidim = np.random.randint(
    0,
    NUM_CLASSES,
    size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
)
_multiclass_preds_multidim = np.random.randint(
    0,
    NUM_CLASSES,
    size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
)
_multiclass_probs_multidim = np.random.rand(
    NUM_BATCHES,
    BATCH_SIZE,
    NUM_CLASSES,
    EXTRA_DIM,
)


def _multiclass_cases(*, xp: Any):
    """Return multiclass input cases for the given array namespace."""
    return (
        pytest.param(
            InputSpec(
                target=xp.asarray(_multiclass_labels_0d),
                preds=xp.asarray(_multiclass_preds_0d),
            ),
            id="input[single-element-labels]",
        ),
        pytest.param(
            InputSpec(
                target=xp.asarray(_multiclass_labels_1d),
                preds=xp.asarray(_multiclass_preds_1d),
            ),
            id="input[1d-labels]",
        ),
        pytest.param(
            InputSpec(
                preds=_multiclass_with_missing_class(
                    NUM_BATCHES,
                    BATCH_SIZE,
                    num_classes=NUM_CLASSES,
                    xp=xp,
                ),
                target=_multiclass_with_missing_class(
                    NUM_BATCHES,
                    BATCH_SIZE,
                    num_classes=NUM_CLASSES,
                    xp=xp,
                ),
            ),
            id="input[1d-labels-missing_class]",
        ),
        pytest.param(
            InputSpec(
                target=xp.asarray(_multiclass_labels_multidim),
                preds=xp.asarray(_multiclass_preds_multidim),
            ),
            id="input[multidim-labels]",
        ),
        pytest.param(
            InputSpec(
                target=xp.asarray(_multiclass_labels_0d),
                preds=xp.asarray(_multiclass_probs_0d),
            ),
            id="input[single-element-probs]",
        ),
        pytest.param(
            InputSpec(
                target=xp.asarray(_multiclass_labels_0d),
                preds=xp.asarray(log_softmax(_multiclass_probs_0d, axis=-1)),
            ),
            id="input[single-element-logits]",
        ),
        pytest.param(
            InputSpec(
                target=xp.asarray(_multiclass_labels_1d),
                preds=xp.asarray(_multiclass_probs_1d),
            ),
            id="input[1d-probs]",
        ),
        pytest.param(
            InputSpec(
                target=xp.asarray(_multiclass_labels_1d),
                preds=xp.asarray(log_softmax(_multiclass_probs_1d, axis=-1)),
            ),
            id="input[1d-logits]",
        ),
        pytest.param(
            InputSpec(
                target=xp.asarray(_multiclass_labels_multidim),
                preds=xp.asarray(_multiclass_probs_multidim),
            ),
            id="input[multidim-probs]",
        ),
        pytest.param(
            InputSpec(
                target=xp.asarray(_multiclass_labels_multidim),
                preds=xp.asarray(log_softmax(_multiclass_probs_multidim, axis=-1)),
            ),
            id="input[multidim-logits]",
        ),
    )


# multilabel
_multilabel_labels = np.random.randint(0, 2, size=(NUM_BATCHES, BATCH_SIZE, NUM_LABELS))
_multilabel_preds = np.random.randint(
    0,
    2,
    size=(NUM_BATCHES, BATCH_SIZE, NUM_LABELS),
)
_multilabel_probs = np.random.rand(NUM_BATCHES, BATCH_SIZE, NUM_LABELS)
_multilabel_labels_multidim = np.random.randint(
    0,
    2,
    size=(NUM_BATCHES, BATCH_SIZE, NUM_LABELS, EXTRA_DIM),
)
_multilabel_preds_multidim = np.random.randint(
    0,
    2,
    size=(NUM_BATCHES, BATCH_SIZE, NUM_LABELS, EXTRA_DIM),
)
_multilabel_probs_multidim = np.random.rand(
    NUM_BATCHES,
    BATCH_SIZE,
    NUM_LABELS,
    EXTRA_DIM,
)


def _multilabel_cases(*, xp: Any):
    return (
        pytest.param(
            InputSpec(
                target=xp.asarray(_multilabel_labels),
                preds=xp.asarray(_multilabel_preds),
            ),
            id="input[2d-labels]",
        ),
        pytest.param(
            InputSpec(
                target=xp.asarray(_multilabel_labels_multidim),
                preds=xp.asarray(_multilabel_preds_multidim),
            ),
            id="input[multidim-labels]",
        ),
        pytest.param(
            InputSpec(
                target=xp.asarray(_multilabel_labels),
                preds=xp.asarray(_multilabel_probs),
            ),
            id="input[2d-probs]",
        ),
        pytest.param(
            InputSpec(
                target=xp.asarray(_multilabel_labels),
                preds=xp.asarray(_inv_sigmoid(_multilabel_probs)),
            ),
            id="input[2d-logits]",
        ),
        pytest.param(
            InputSpec(
                target=xp.asarray(_multilabel_labels_multidim),
                preds=xp.asarray(_multilabel_probs_multidim),
            ),
            id="input[multidim-probs]",
        ),
        pytest.param(
            InputSpec(
                target=xp.asarray(_multilabel_labels_multidim),
                preds=xp.asarray(_inv_sigmoid(_multilabel_probs_multidim)),
            ),
            id="input[multidim-logits]",
        ),
    )
