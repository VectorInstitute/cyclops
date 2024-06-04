"""Collection of metrics."""

import hashlib
import itertools
import json
import logging
import warnings
from collections import OrderedDict, UserDict
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import array_api_compat as apc
import numpy as np

from cyclops.evaluate.metrics.experimental.metric import Metric
from cyclops.evaluate.metrics.experimental.utils.ops import clone
from cyclops.evaluate.metrics.experimental.utils.types import Array
from cyclops.utils.log import setup_logging
from cyclops.utils.optional import import_optional_module


if TYPE_CHECKING:
    from torchmetrics.metric import Metric as TorchMetric
else:
    TorchMetric = import_optional_module(
        "torchmetrics.metric",
        attribute="Metric",
        error="ignore",
    )
    if TorchMetric is None:
        TorchMetric = type(None)


LOGGER = logging.getLogger(__name__)
setup_logging(print_level="WARN", logger=LOGGER)


class ArrayEncoder(json.JSONEncoder):
    """A custom JSON encoder for objects conforming to the array API standard."""

    def default(self, obj: Any) -> Any:
        """Return a JSON-serializable representation of the object.

        Objects conforming to the array API standard are converted to Python lists
        via numpy. Arrays are moved to the CPU before converting to numpy.
        """
        if apc.is_array_api_obj(obj):
            return np.from_dlpack(apc.to_device(obj, "cpu")).tolist()
        return json.JSONEncoder.default(self, obj)


class MetricDict(UserDict[str, Union[Metric, TorchMetric]]):
    """A dictionary-like object for grouping metrics and computing them together.

    This class is used to group metrics together. It is useful for when you want
    to compute multiple metrics at the same time. It behaves like a dictionary
    where the keys are the names of the metrics and the values are the metrics
    themselves. Internally, it groups metrics with similar states together to
    reduce the number of times the state is updated.

    Parameters
    ----------
    metrics : Union[Metric, Sequence[Metric], Dict[str, Metric]], optional, default=None
        The metrics to add to the dictionary. This can be a single metric, a
        sequence of metrics, or a dictionary mapping names to metrics.
    *other_metrics : Metric, optional
        Additional metrics to add to the dictionary. The metric will be added with
        the name of the class of the metric as the key. This is only used if
        `metrics` is a single metric or a sequence of metrics.
    prefix : str, optional, default=None
        A prefix to add to the names of the metrics.
    postfix : str, optional, default=None
        A postfix to add to the names of the metrics.
    **kwargs : Metric, optional
        Additional metrics to add to the dictionary as keyword arguments.

    Raises
    ------
    TypeError
        If `metrics` is not a metric, a sequence of containing at least one metric,
        or a dictionary mapping at least one metric name to a metric object.
    TypeError
        If `other_metrics` is not empty and `metrics` is not a single metric or a
        sequence of metrics.
    TypeError
        If `prefix` or `postfix` is not a string.

    Warnings
    --------
    While this class can be used with objects of type `torchmetrics.metric.Metric`,
    users to be weary of the following caveats:
    - The `update` and `__call__` methods of `torchmetrics.metric.Metric` expects
    the first and second positional arguments to be the `preds` and `targets`,
    respectively. This is the opposite of the `Metric` class in this module.
    To get around this issue, always use keyword arguments when calling the
    `update` and `__call__` of this object with `torchmetrics.metric.Metric` objects.
    - Mixing `torchmetrics.metric.Metric` objects with `Metric` objects in the same
    collection would restrict the array inputs to be of type `torch.Tensor`.

    Examples
    --------
    >>> from cyclops.evaluate.metrics.experimental import MetricDict
    >>> from cyclops.evaluate.metrics.experimental import (
    ...     BinaryAccuracy,
    ...     BinaryF1Score,
    ...     BinaryPrecision,
    ...     BinaryRecall,
    ... )
    >>> import numpy.array_api as anp
    >>> target = anp.asarray([0, 1, 0, 1])
    >>> preds = anp.asarray([0, 1, 1, 1])
    >>> metric_dict = MetricDict(BinaryAccuracy(), BinaryF1Score())
    >>> metric_dict(target, preds)
    {'BinaryAccuracy': Array(0.75, dtype=float32), 'BinaryF1Score': Array(0.8, dtype=float32)}
    >>> metric_dict.reset()
    >>> metric_dict.add_metrics(BinaryPrecision(), BinaryRecall())
    >>> metric_dict(target, preds)
    {'BinaryAccuracy': Array(0.75, dtype=float32), 'BinaryF1Score': Array(0.8, dtype=float32), 'BinaryPrecision': Array(0.6666667, dtype=float32), 'BinaryRecall': Array(1., dtype=float32)}

    """  # noqa: W505

    _metric_groups: Dict[int, List[str]]

    def __init__(
        self,
        metrics: Optional[
            Union[
                Metric,
                TorchMetric,
                Sequence[Union[Metric, TorchMetric]],
                Dict[str, Union[Metric, TorchMetric]],
            ]
        ] = None,
        *other_metrics: Union[Metric, TorchMetric],
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
        **kwargs: Union[Metric, TorchMetric],
    ) -> None:
        """Initialize the metric collection."""
        super().__init__()

        self._validate_adfix(prefix, postfix)
        self.prefix = prefix
        self.postfix = postfix
        self._groups_created: bool = False
        self._state_is_copy: bool = False

        self.add_metrics(metrics, *other_metrics, **kwargs)

    def _validate_adfix(self, prefix: Optional[str], postfix: Optional[str]) -> None:
        """Check that the arguments `prefix` and `postfix` are strings."""
        if prefix is not None and not isinstance(prefix, str):
            raise TypeError(
                f"Expected `prefix` to be a string, but got {type(prefix).__name__}.",
            )
        if postfix is not None and not isinstance(postfix, str):
            raise TypeError(
                f"Expected `postfix` to be a string, but got {type(postfix).__name__}.",
            )

    def _validate_metric_arg(
        self,
        metrics: Optional[
            Union[
                Metric,
                TorchMetric,
                Sequence[Union[Metric, TorchMetric]],
                Dict[str, Union[Metric, TorchMetric]],
            ]
        ] = None,
        *other_metrics: Union[Metric, TorchMetric],
        **kwargs: Union[Metric, TorchMetric],
    ) -> None:
        """Check that the arguments `metrics`, `other_metrics`, and `kwargs`."""
        if isinstance(metrics, Sequence) and not any(
            isinstance(metric, (Metric, TorchMetric)) for metric in metrics
        ):
            raise TypeError(
                "Expected `metrics` to be a sequence containing at least one "
                "metric object, but got either an empty sequence or a sequence "
                f"containing non-metric objects: {metrics}.",
            )
        if isinstance(metrics, dict) and not any(
            isinstance(m_k, str) and isinstance(m_v, (Metric, TorchMetric))
            for m_k, m_v in metrics.items()
        ):
            raise TypeError(
                "Expected `metrics` to be a dictionary mapping metric names to "
                "metric objects, but got an empty dictionary or a dictionary "
                "containing non-metric objects or a dictionary with non-string "
                f"keys: {metrics}.",
            )
        if metrics is not None and not isinstance(
            metrics,
            (Sequence, dict, Metric, TorchMetric),
        ):
            raise TypeError(
                f"Expected `metrics` to be of type `Metric`, `Sequence[Metric]` or "
                f"`Dict[str, Metric]`, but got {type(metrics)}.",
            )

        if (
            other_metrics
            and metrics is not None
            and not isinstance(metrics, (Metric, TorchMetric, Sequence))
        ):
            raise TypeError(
                "The argument `other_metrics` can only be used if `metrics` is a "
                "single metric or a sequence of metrics.",
            )
        if other_metrics and not any(
            isinstance(metric, (Metric, TorchMetric)) for metric in other_metrics
        ):
            raise TypeError(
                "Expected `other_metrics` to be a sequence containing at least one "
                "metric object, but got either an empty sequence or a sequence "
                f"containing non-metric objects: {other_metrics}.",
            )
        if kwargs and not any(
            isinstance(metric, (Metric, TorchMetric)) for metric in kwargs.values()
        ):
            raise TypeError(
                "Expected `kwargs` to contain at least one metric object, but found "
                f"only non-metric objects: {kwargs}.",
            )

    def _create_metric_groups(self) -> None:
        """Group metrics with similar states together.

        Notes
        -----
        This method uses a hashing function on the serialized state of each metric
        to group metrics with similar states together.

        """
        metrics_by_state: Dict[str, List[str]] = {}
        for name, metric in self.data.items():
            state_hash = hashlib.md5(
                json.dumps(metric._defaults, cls=ArrayEncoder, sort_keys=True).encode(),
            ).hexdigest()
            metrics_by_state.setdefault(state_hash, []).append(name)

        self._metric_groups = dict(zip(itertools.count(), metrics_by_state.values()))

    def add_metrics(  # noqa: PLR0912
        self,
        metrics: Optional[
            Union[
                Metric,
                TorchMetric,
                Sequence[Union[Metric, TorchMetric]],
                Dict[str, Union[Metric, TorchMetric]],
            ]
        ] = None,
        *other_metrics: Union[Metric, TorchMetric],
        **kwargs: Union[Metric, TorchMetric],
    ) -> None:
        """Add metrics to the dictionary.

        Parameters
        ----------
        metrics : Union[Metric, Sequence[Metric], Dict[str, Metric]], optional
            The metrics to add to the dictionary. This can be a single metric, a
            sequence of metrics, or a dictionary mapping names to metrics.
        *other_metrics : Metric, optional
            Additional metrics to add to the dictionary. The metric will be added
            with the name of the class of the metric as the key. This is only used
            if `metrics` is a single metric or a sequence of metrics.
        **kwargs : Metric, optional
            Additional metrics to add to the dictionary.

        Raises
        ------
        TypeError
            If `metrics` is not a metric, a sequence of containing at least one metric,
            or a dictionary mapping at least one metric name to a metric object.
        TypeError
            If `other_metrics` is not empty and `metrics` is not a single metric or a
            sequence of metrics.
        TypeError
            If `prefix` or `postfix` is not a string.
        """
        if metrics is None and not other_metrics and not kwargs:
            return

        self._validate_metric_arg(metrics, *other_metrics, **kwargs)

        def get_warning_msg(arg_name: str, obj: Any) -> str:
            """Return a warning message for invalid objects."""
            return (
                f"Found object in `{arg_name}` that is not `Metric` or `TorchMetric`. "
                f"This object will be ignored: {obj}."
            )

        if isinstance(metrics, (Metric, TorchMetric)):
            metrics = [metrics]

        if isinstance(metrics, Sequence):
            for metric in metrics:
                if isinstance(metric, (Metric, TorchMetric)):
                    self.data[metric.__class__.__name__] = metric
                else:
                    warnings.warn(
                        get_warning_msg("metrics", metric),
                        category=UserWarning,
                        stacklevel=1,
                    )
        elif isinstance(metrics, dict):
            for name, metric in metrics.items():
                if isinstance(metric, (Metric, TorchMetric)):
                    self.data[name] = metric
                else:
                    warnings.warn(
                        get_warning_msg("metrics", metric),
                        category=UserWarning,
                        stacklevel=1,
                    )
        for metric in other_metrics:
            if isinstance(metric, (Metric, TorchMetric)):
                self.data[metric.__class__.__name__] = metric
            else:
                warnings.warn(
                    get_warning_msg("other_metrics", metric),
                    category=UserWarning,
                    stacklevel=1,
                )
        for name, metric in kwargs.items():
            if isinstance(metric, (Metric, TorchMetric)):
                self.data[name] = metric
            else:
                warnings.warn(
                    get_warning_msg("kwargs", metric),
                    category=UserWarning,
                    stacklevel=1,
                )

        if self._groups_created:
            self._create_metric_groups()  # update the groups

    def _create_metric_groups_state_ref(self, copy_state: bool = False) -> None:
        """Create references between metrics in the same group."""

        def deepcopy_state(obj: Any) -> Any:
            """Deepcopy a state variable of a metric."""
            if apc.is_array_api_obj(obj):
                return clone(obj)
            return deepcopy(obj)

        if self._groups_created and not self._state_is_copy:
            for metric_names in self._metric_groups.values():
                base_metric = self.data[metric_names[0]]
                for metric_name in metric_names[1:]:
                    for state in base_metric._defaults:
                        base_metric_state = getattr(base_metric, state)
                        setattr(
                            self.data[metric_name],
                            state,
                            deepcopy_state(base_metric_state)
                            if copy_state
                            else base_metric_state,
                        )

                    self.data[metric_name]._update_count = (
                        deepcopy(base_metric._update_count)
                        if copy_state
                        else base_metric._update_count
                    )
        self._state_is_copy = copy_state

    def _set_name(self, base: str) -> str:
        """Adjust name of  metric with both prefix and postfix."""
        name = base if self.prefix is None else self.prefix + base
        return name if self.postfix is None else name + self.postfix

    def _to_renamed_ordered_dict(self) -> OrderedDict[str, Metric]:
        """Return an ordered dict with the renamed keys."""
        ordered_data = OrderedDict()
        for key, value in self.data.items():
            ordered_data[self._set_name(key)] = value
        return ordered_data  # type: ignore

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the state of all metrics in the collection.

        Uses the metric groups to only update the state variables once per group.

        Parameters
        ----------
        *args: Any
            The positional arguments to pass to the update_state method of each metric.
        **kwargs: Any
            The keyword arguments to pass to the update_state method of each metric.

        Warnings
        --------
        When mixing `torchmetrics.metric.Metric` objects with `Metric` objects in
        the same collection, provide the `preds` and `targets` as keyword arguments,
        otherwise the positional arguments will be reversed.

        """
        if self._groups_created:
            # call `update` once per metric group
            for metrics in self._metric_groups.values():
                base_metric = self.data[metrics[0]]
                base_metric.update(*args, **kwargs)

            if self._state_is_copy:
                # if the state is a copy, we need to update the references
                self._create_metric_groups_state_ref()
                self._state_is_copy = False
        else:
            # call `update` separately for each metric to ensure that the state
            # variables are created correctly
            for metric in self.data.values():
                metric.update(*args, **kwargs)

            self._create_metric_groups()
            self._create_metric_groups_state_ref()
            self._groups_created = True

    def compute(self) -> Dict[str, Array]:
        """Compute the metrics in the dictionary."""
        result = {
            name: metric.compute()
            for name, metric in self.items(keep_base=True, copy_state=False)
        }
        result = _flatten_dict(result)
        return {self._set_name(k): v for k, v in result.items()}

    def reset(self) -> None:
        """Reset the state of all metrics in the dictionary."""
        for metric in self.values(copy_state=False):
            metric.reset()

        if self._groups_created:
            self._create_metric_groups_state_ref()  # reset the references

    def keys(self, keep_base: bool = False) -> Iterable[Hashable]:  # type: ignore
        """Return an iterable of the dictionary's keys.

        Parameters
        ----------
        keep_base : bool
            Whether to add prefix/postfix to the keys of items in the dictionary.

        """
        if keep_base:
            return self.data.keys()
        return self._to_renamed_ordered_dict().keys()

    def values(  # type: ignore[override]
        self,
        copy_state: bool = True,
    ) -> Iterable[Metric]:
        """Return an iterable of the underlying dictionary's values.

        Parameters
        ----------
        copy_state : bool, default=True
            Whether to copy the state variables or use references between metrics
            in the same group.

        """
        self._create_metric_groups_state_ref(copy_state)  # update references
        return self.data.values()  # type: ignore

    def items(  # type: ignore[override]
        self,
        keep_base: bool = False,
        copy_state: bool = True,
    ) -> Iterable[Tuple[str, Metric]]:
        """Return an iterable of the underlying dictionary's items.

        Parameters
        ----------
        keep_base : bool
            Whether to add the adfixes to the keys of items in the dictionary.
        copy_state : bool, default=True
            Whether to copy the state variables or use references between metrics
            in the same group.

        """
        self._create_metric_groups_state_ref(copy_state)  # update references
        if keep_base:
            return self.data.items()  # type: ignore
        return self._to_renamed_ordered_dict().items()

    def clone(
        self,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
    ) -> "MetricDict":
        """Create a copy of the metric collection.

        Parameters
        ----------
        prefix : str, optional, default=None
            Prefix to add to the name of the metric.
        postfix : str, optional, default=None
            Postfix to add to the name of the metric.

        Returns
        -------
        MetricDict
            A deep copy of the dictionary.

        """
        new_obj = deepcopy(self)
        self._validate_adfix(prefix, postfix)
        if prefix:
            new_obj.prefix = prefix
        if postfix:
            new_obj.postfix = postfix
        return new_obj

    def to_device(
        self,
        device: str,
        stream: Optional[Union[int, Any]] = None,
    ) -> "MetricDict":
        """Move all metrics to the given device.

        Parameters
        ----------
        device : str
            The device to move the metrics to.
        stream : int or stream, optional
            The stream to move the metrics to.

        Returns
        -------
        MetricDict
            The dictionary with all metrics moved to the given device.

        """
        for metric in self.values(copy_state=False):
            if isinstance(metric, TorchMetric):
                metric.to(device)
            else:
                metric.to_device(device, stream)
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Array]:
        """Apply the __call__ method of all metrics in the collection."""
        batch_result = {
            name: metric(*args, **kwargs) for name, metric in self.items(keep_base=True)
        }
        batch_result = _flatten_dict(batch_result)
        return {self._set_name(k): v for k, v in batch_result.items()}

    def __getitem__(self, key: str, copy_state: bool = True) -> Metric:
        """Return the metric with the given key."""
        self._create_metric_groups_state_ref(copy_state)  # update references
        return self.data[key]  # type: ignore

    def __iter__(self, keep_base: bool = False) -> Iterable[str]:  # type: ignore[override]
        """Return an iterable of the dictionary's keys."""
        if keep_base:
            return iter(self.data)
        return iter(self._to_renamed_ordered_dict())


def _flatten_dict(a_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten dict of dicts into single dict."""
    new_dict = {}
    for key, value in a_dict.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                new_dict[sub_key] = sub_value
        else:
            new_dict[key] = value
    return new_dict
