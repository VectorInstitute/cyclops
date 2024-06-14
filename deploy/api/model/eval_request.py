"""A Modle for requesting evaluation."""
from typing import List, Union

from pydantic import BaseModel


class RequestData(BaseModel):
    """A Modle for requesting evaluation."""

    target: List[Union[int, float]]
    preds_prob: List[float]
    metric_slice: str | None = None
    metrics: List[str] | None = None
