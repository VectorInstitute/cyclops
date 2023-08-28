"""Factory for creating metrics."""

from difflib import get_close_matches
from typing import Any, List, Mapping, Optional

from cyclops.evaluate.metrics.metric import _METRIC_REGISTRY, Metric


def create_metric(metric_name: str, **kwargs: Optional[Mapping[str, Any]]) -> Metric:
    """Create a metric instance from a name.

    Parameters
    ----------
    metric_name : str
        The name of the metric.
    **kwargs : Mapping[str, Any], optional
        The keyword arguments to pass to the metric constructor.

    Returns
    -------
    metric : Metric
        The metric instance.

    """
    metric_class = _METRIC_REGISTRY.get(metric_name, None)
    if metric_class is None:
        similar_keys_list: List[str] = get_close_matches(
            metric_name,
            _METRIC_REGISTRY.keys(),
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
