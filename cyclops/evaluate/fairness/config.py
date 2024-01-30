"""Configuration for fairness evaluator."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from datasets import Dataset, config

from cyclops.evaluate.metrics.experimental.metric import Metric
from cyclops.evaluate.metrics.experimental.metric_dict import MetricDict


@dataclass
class FairnessConfig:
    """Configuration for fairness metrics."""

    metrics: Union[str, Callable[..., Any], Metric, MetricDict]
    dataset: Dataset
    groups: Union[str, List[str]]
    target_columns: Union[str, List[str]]
    prediction_columns: Union[str, List[str]] = "predictions"
    group_values: Optional[Dict[str, Any]] = None
    group_bins: Optional[Dict[str, Union[int, List[Any]]]] = None
    group_base_values: Optional[Dict[str, Any]] = None
    thresholds: Optional[Union[int, List[float]]] = None
    compute_optimal_threshold: bool = False
    remove_columns: Optional[Union[str, List[str]]] = None
    batch_size: Optional[int] = config.DEFAULT_MAX_BATCH_SIZE
    metric_name: Optional[str] = None
    metric_kwargs: Optional[Dict[str, Any]] = None
