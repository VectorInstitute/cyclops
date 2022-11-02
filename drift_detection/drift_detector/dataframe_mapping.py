from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DataFrameMapping:
    id: Optional[str]
    timestamp: Optional[str] = None
    targets: Optional[List[str]] = None
    predictions: Optional[List[str]] = None
    numericals: Optional[List[str]] = None
    categoricals: Optional[List[str]] = None
    gender: Optional[str] = None
