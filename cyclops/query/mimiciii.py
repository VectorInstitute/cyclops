"""MIMIC-III query API.

Supports querying of MIMIC-III.

"""

import logging
from typing import Any, Dict, Optional

import cyclops.query.ops as qo
from cyclops.query.base import DatasetQuerier
from cyclops.query.interface import QueryInterface
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


# Constants.
PATIENTS = "patients"
ADMISSIONS = "admissions"
DIAGNOSES_ICD = "diagnoses_icd"
LABEVENTS = "labevents"
CHARTEVENTS = "chartevents"
TRANSFERS = "transfers"
PRESCRIPTIONS = "prescriptions"
MICROBIOLOGYEVENTS = "microbiologyevents"
PROCEDUREEVENTS_MV = "procedureevents_mv"
DIM_ITEMS = "dim_items"
DIM_LABITEMS = "dim_labitems"
DIM_ICD_DIAGNOSES = "dim_icd_diagnoses"


TABLE_MAP = {
    PATIENTS: lambda db: db.mimiciii.patients,
    ADMISSIONS: lambda db: db.mimiciii.admissions,
    DIAGNOSES_ICD: lambda db: db.mimiciii.diagnoses_icd,
    LABEVENTS: lambda db: db.mimiciii.labevents,
    CHARTEVENTS: lambda db: db.mimiciii.chartevents,
    TRANSFERS: lambda db: db.mimiciii.transfers,
    PRESCRIPTIONS: lambda db: db.mimiciii.prescriptions,
    MICROBIOLOGYEVENTS: lambda db: db.mimiciii.microbiologyevents,
    PROCEDUREEVENTS_MV: lambda db: db.mimiciii.procedureevents_mv,
    DIM_LABITEMS: lambda db: db.mimiciii.d_labitems,
    DIM_ITEMS: lambda db: db.mimiciii.d_items,
    DIM_ICD_DIAGNOSES: lambda db: db.mimiciii.d_icd_diagnoses,
}


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
        super().__init__(TABLE_MAP, **overrides)

    def diagnoses_icd(
        self,
        join: Optional[qo.JoinArgs] = None,
        ops: Optional[qo.Sequential] = None,
    ) -> QueryInterface:
        """Query MIMICIII diagnosis data.

        Parameters
        ----------
        join: qo.JoinArgs, optional
            Join arguments.
        ops: qo.Sequential, optional
            Additional operations to apply to the query.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed query, wrapped in an interface object.

        """
        table = self.get_table(DIAGNOSES_ICD)

        # Join with diagnoses dimension table.
        table = qo.Join(
            join_table=self.get_table(DIM_ICD_DIAGNOSES),
            on=["icd9_code"],
            on_to_type=["str"],
        )(table)

        return QueryInterface(self._db, table, join=join, ops=ops)

    def labevents(
        self,
        join: Optional[qo.JoinArgs] = None,
        ops: Optional[qo.Sequential] = None,
    ) -> QueryInterface:
        """Query MIMICIII labevents data.

        Parameters
        ----------
        join: qo.JoinArgs, optional
            Join arguments.
        ops: qo.Sequential, optional
            Additional operations to apply to the query.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed query, wrapped in an interface object.

        """
        table = self.get_table(LABEVENTS)

        # Join with lab dimension table.
        table = qo.Join(
            join_table=self.get_table(DIM_LABITEMS),
            on=["itemid"],
            on_to_type=["str"],
        )(table)

        return QueryInterface(self._db, table, join=join, ops=ops)

    def chartevents(
        self,
        join: Optional[qo.JoinArgs] = None,
        ops: Optional[qo.Sequential] = None,
    ) -> QueryInterface:
        """Query MIMICIII chartevents data.

        Parameters
        ----------
        join: qo.JoinArgs, optional
            Join arguments.
        ops: qo.Sequential, optional
            Additional operations to apply to the query.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed query, wrapped in an interface object.

        """
        table = self.get_table(CHARTEVENTS)

        # Join with dimension table.
        table = qo.Join(
            join_table=self.get_table(DIM_ITEMS),
            on=["itemid"],
            on_to_type=["str"],
        )(table)

        return QueryInterface(self._db, table, join=join, ops=ops)
