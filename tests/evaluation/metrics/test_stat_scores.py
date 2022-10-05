import pytest
import numpy as np
from sklearn.utils._testing import assert_array_equal
from evaluation.metrics.functional.stat_scores import (
    binary_stat_scores,
    multiclass_stat_scores,
    multilabel_stat_scores,
)


@pytest.fixture
def binary_cases():
    """Binary cases."""
    int_preds = [1, 1, 1, 0, 0, 0]
    target = [0, 1, 0, 1, 0, 1]
    proba_preds = [0.8, 0.55, 0.75, 0.38, 0.49, 0.22]

    return int_preds, proba_preds, target


@pytest.fixture
def multiclass_cases():
    """Multiclass cases."""
    int_preds = [0, 0, 2, 2, 0, 2]
    target = [2, 0, 2, 2, 0, 1]

    proba_preds = [
        [0.9, 0.1, 0],
        [0.6, 0.1, 0.3],
        [0.2, 0.3, 0.5],
        [0.1, 0, 0.9],
        [0.6, 0.1, 0.3],
        [0.3, 0.2, 0.5],
    ]

    return int_preds, proba_preds, target


@pytest.fixture
def multilabel_cases():
    """Multilabel cases."""
    int_preds = [[0, 0, 1], [1, 0, 1]]
    target = [[0, 1, 0], [1, 0, 1]]

    proba_preds = [[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]]

    return int_preds, proba_preds, target


@pytest.mark.parametrize("threshold", [0.5, 0.1, 0.9])
def test_binary_stat_scores(binary_cases, threshold):
    """Test binary case."""
    int_preds, proba_preds, target = binary_cases

    # same preds
    same_input_result = [3, 0, 3, 0, 3]
    assert_array_equal(binary_stat_scores(int_preds, int_preds), same_input_result)

    result = [1, 2, 1, 2, 3]

    # preds: int list, target: int list
    output = binary_stat_scores(target, int_preds)
    assert_array_equal(output, result)

    # probablistic preds
    if threshold == 0.1:
        result = [3, 3, 0, 0, 3]
    elif threshold == 0.9:
        result = [0, 0, 3, 3, 3]

    # preds: list of probability scores, target: int list
    output = binary_stat_scores(target, proba_preds, threshold=threshold)
    assert_array_equal(output, result)


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_multiclass_stat_scores(multiclass_cases, top_k):
    """Test multiclass case."""
    int_preds, proba_preds, target = multiclass_cases

    # same preds
    assert_array_equal(
        multiclass_stat_scores(int_preds, int_preds, num_classes=3), [6, 0, 12, 0, 6]
    )
    assert_array_equal(
        multiclass_stat_scores(int_preds, int_preds, num_classes=3, classwise=True),
        [[3, 0, 3, 0, 3], [0, 0, 6, 0, 0], [3, 0, 3, 0, 3]],
    )

    result_global = [4, 2, 10, 2, 6]
    result_classwise = [[2, 1, 3, 0, 2], [0, 0, 5, 1, 1], [2, 1, 2, 1, 3]]

    output = multiclass_stat_scores(target, int_preds, num_classes=3)
    assert_array_equal(output, result_global)

    output = multiclass_stat_scores(target, int_preds, num_classes=3, classwise=True)
    assert_array_equal(output, result_classwise)

    # probablistic preds
    result_global = [4, 2, 10, 2, 6]
    result_classwise = [[2, 1, 3, 0, 2], [0, 0, 5, 1, 1], [2, 1, 2, 1, 3]]

    # top_k
    if top_k == 2:
        result_global = [4, 8, 4, 2, 6]
        result_classwise = [[2, 3, 1, 0, 2], [0, 2, 3, 1, 1], [2, 3, 0, 1, 3]]
    if top_k == 3:
        with pytest.raises(
            ValueError,
            match=r"The `top_k` has to be strictly smaller than the number of classes.",
        ):
            multiclass_stat_scores(target, proba_preds, num_classes=3, top_k=top_k)
    else:
        output = multiclass_stat_scores(
            target,
            proba_preds,
            num_classes=3,
            top_k=top_k,
        )
        assert_array_equal(output, result_global)

        output = multiclass_stat_scores(
            target, proba_preds, num_classes=3, classwise=True, top_k=top_k
        )
        assert_array_equal(output, result_classwise)


def test_multilabel_stat_scores(multilabel_cases):
    """Test multilabel case."""
    int_preds, proba_preds, target = multilabel_cases

    # same preds
    assert_array_equal(
        multilabel_stat_scores(int_preds, int_preds, num_labels=3), [3, 0, 3, 0, 3]
    )
    assert_array_equal(
        multilabel_stat_scores(int_preds, int_preds, num_labels=3, reduce="macro"),
        [[1, 0, 1, 0, 1], [0, 0, 2, 0, 0], [2, 0, 0, 0, 2]],
    )
    assert_array_equal(
        multilabel_stat_scores(int_preds, int_preds, num_labels=3, reduce="samples"),
        [[1, 0, 2, 0, 1], [2, 0, 1, 0, 2]],
    )

    result_micro = [2, 1, 2, 1, 3]
    result_macro = [[1, 0, 1, 0, 1], [0, 0, 1, 1, 1], [1, 1, 0, 0, 1]]
    result_samples = [[0, 1, 1, 1, 1], [2, 0, 1, 0, 2]]

    output = multilabel_stat_scores(target, int_preds, num_labels=3)
    assert_array_equal(output, result_micro)

    output = multilabel_stat_scores(target, int_preds, num_labels=3, reduce="macro")
    assert_array_equal(output, result_macro)

    output = multilabel_stat_scores(target, int_preds, num_labels=3, reduce="samples")
    assert_array_equal(output, result_samples)

    # probablistic preds
    output = multilabel_stat_scores(target, proba_preds, num_labels=3)
    assert_array_equal(output, result_micro)

    output = multilabel_stat_scores(target, proba_preds, num_labels=3, reduce="macro")
    assert_array_equal(output, result_macro)

    output = multilabel_stat_scores(target, proba_preds, num_labels=3, reduce="samples")
    assert_array_equal(output, result_samples)

    # top_k
    result_micro = [3, 1, 2, 0, 3]
    result_macro = [[1, 0, 1, 0, 1], [1, 0, 1, 0, 1], [1, 1, 0, 0, 1]]
    result_samples = [[1, 1, 1, 0, 1], [2, 0, 1, 0, 2]]

    output = multilabel_stat_scores(target, proba_preds, num_labels=3, top_k=2)
    assert_array_equal(output, result_micro)

    output = multilabel_stat_scores(
        target, proba_preds, num_labels=3, reduce="macro", top_k=2
    )
    assert_array_equal(output, result_macro)

    output = multilabel_stat_scores(
        target, proba_preds, num_labels=3, reduce="samples", top_k=2
    )
    assert_array_equal(output, result_samples)

    # top_k > num_labels
    with pytest.raises(
        ValueError,
        match=r"The `top_k` has to be strictly smaller than the number of classes.",
    ):
        multilabel_stat_scores(target, proba_preds, num_labels=3, top_k=4)

    # top_k = num_labels
    with pytest.raises(
        ValueError,
        match=r"The `top_k` has to be strictly smaller than the number of classes.",
    ):
        multilabel_stat_scores(target, proba_preds, num_labels=3, top_k=3)
