"""Factory for creating metrics."""

from difflib import get_close_matches
from typing import Any, List, Union

from cyclops.evaluate.metrics.experimental.metric import (
    _METRIC_REGISTRY as _EXPERIMENTAL_METRIC_REGISTRY,
)
from cyclops.evaluate.metrics.experimental.metric import Metric as ExperimentalMetric
from cyclops.evaluate.metrics.metric import _METRIC_REGISTRY, Metric


def create_metric(
    metric_name: str,
    experimental: bool = False,
    **kwargs: Any,
) -> Union[Metric, ExperimentalMetric]:
    """Create a metric instance from a name.

    Parameters
    ----------
    metric_name : str
        The name of the metric.
    experimental : bool
        Whether to use metrics from `cyclops.evaluate.metrics.experimental`.
    **kwargs : Any
        The keyword arguments to pass to the metric constructor.

    Returns
    -------
    metric : Metric
        The metric instance.

    """
    metric_class = (
        _METRIC_REGISTRY.get(metric_name, None)
        if not experimental
        else _EXPERIMENTAL_METRIC_REGISTRY.get(metric_name, None)
    )
    if metric_class is None:
        registry_keys: List[str] = (
            list(_METRIC_REGISTRY.keys())
            if not experimental
            else list(_EXPERIMENTAL_METRIC_REGISTRY.keys())  # type: ignore[arg-type]
        )
        similar_keys_list: List[str] = get_close_matches(
            metric_name,
            registry_keys,
            n=5,
        )
        similar_keys: str = ", ".join(similar_keys_list)
        similar_keys = (
            f" Did you mean one of: {similar_keys}?"
            if similar_keys
            else " It may not be registered."
        )
        raise ValueError(f"Metric {metric_name} not found.{similar_keys}")

    metric: Metric = metric_class(**kwargs)

    return metric
