"""EICU-CRD query API.

Supports querying of eICU.

"""

import logging

from cyclops.query.base import DatasetQuerier
from cyclops.utils.log import setup_logging


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


class EICUQuerier(DatasetQuerier):
    """EICU dataset querier."""
