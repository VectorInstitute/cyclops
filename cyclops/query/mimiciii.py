"""MIMIC-III query API.

Supports querying of MIMIC-III.

"""

import logging
from typing import Any, Dict

import cyclops.query.ops as qo
from cyclops.query.base import DatasetQuerier
from cyclops.query.interface import QueryInterface
from cyclops.utils.log import setup_logging


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


class MIMICIIIQuerier(DatasetQuerier):
    """MIMIC-III dataset querier."""

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

    def diagnoses(
        self,
    ) -> QueryInterface:
        """Query MIMICIII diagnosis data.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed query, wrapped in an interface object.

        """
        table = self.get_table("mimiciii", "diagnoses_icd")

        # Join with diagnoses dimension table.
        table = qo.Join(
            join_table=self.get_table("mimiciii", "d_icd_diagnoses"),
            on=["icd9_code"],
            on_to_type=["str"],
        )(table)

        return QueryInterface(self.db, table)

    def labevents(
        self,
    ) -> QueryInterface:
        """Query MIMICIII labevents data.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed query, wrapped in an interface object.

        """
        table = self.get_table("mimiciii", "labevents")

        # Join with lab dimension table.
        table = qo.Join(
            join_table=self.get_table("mimiciii", "d_labitems"),
            on=["itemid"],
            on_to_type=["str"],
        )(table)

        return QueryInterface(self.db, table)

    def chartevents(
        self,
    ) -> QueryInterface:
        """Query MIMICIII chartevents data.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed query, wrapped in an interface object.

        """
        table = self.get_table("mimiciii", "chartevents")

        # Join with dimension table.
        table = qo.Join(
            join_table=self.get_table("mimiciii", "d_items"),
            on=["itemid"],
            on_to_type=["str"],
        )(table)

        return QueryInterface(self.db, table)
