"""eICU-CRD query API.

Supports querying of eICU.

"""

import logging
from typing import Any, Dict

from cyclops.query.base import DatasetQuerier
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


class EICUQuerier(DatasetQuerier):
    """eICU dataset querier."""

    def __init__(self, **config_overrides: Dict[str, Any]) -> None:
        """Initialize.

        Parameters
        ----------
        **config_overrides
            Override configuration parameters, specified as kwargs.

        """
        overrides = {}
        if config_overrides:
            overrides = config_overrides
        super().__init__(**overrides)
