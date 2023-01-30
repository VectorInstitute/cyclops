"""MIMIC-IV query API.

Supports querying of MIMICIV-2.0.

"""

import logging
from typing import Optional

from sqlalchemy import Integer, func, select

import cyclops.query.ops as qo
from cyclops.query.base import DatasetQuerier
from cyclops.query.interface import QueryInterface, QueryInterfaceProcessed
from cyclops.query.post_process.mimiciv import process_mimic_care_units
from cyclops.query.util import get_column
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


# Constants.
PATIENTS = "patients"
ADMISSIONS = "admissions"
DIAGNOSES = "diagnoses"
DIM_DIAGNOSES = "dim_diagnoses"
DIM_ITEMS = "dim_items"
DIM_LABITEMS = "dim_labitems"
CHARTEVENTS = "chartevents"
TRANSFERS = "transfers"
PHARMACY = "pharmacy"
LABEVENTS = "labevents"
EDSTAYS = "ed_stays"

TABLE_MAP = {
    PATIENTS: lambda db: db.mimiciv_hosp.patients,
    ADMISSIONS: lambda db: db.mimiciv_hosp.admissions,
    DIAGNOSES: lambda db: db.mimiciv_hosp.diagnoses_icd,
    DIM_DIAGNOSES: lambda db: db.mimiciv_hosp.d_icd_diagnoses,
    DIM_LABITEMS: lambda db: db.mimiciv_hosp.d_labitems,
    DIM_ITEMS: lambda db: db.mimiciv_icu.d_items,
    CHARTEVENTS: lambda db: db.mimiciv_icu.chartevents,
    LABEVENTS: lambda db: db.mimiciv_hosp.labevents,
    PHARMACY: lambda db: db.mimiciv_hosp.pharmacy,
    TRANSFERS: lambda db: db.mimiciv_hosp.transfers,
    EDSTAYS: lambda db: db.mimic_ed.edstays,
}


class MIMICIVQuerier(DatasetQuerier):
    """MIMICIV dataset querier."""

    def __init__(self, **config_overrides):
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

    def patients(
        self,
        join: Optional[qo.JoinArgs] = None,
        ops: Optional[qo.Sequential] = None,
    ) -> QueryInterface:
        """Query MIMIC patient data.

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

        Notes
        -----
        The function infers the approximate year a patient received care, using the
        `anchor_year` and `anchor_year_group` columns. The `join` and `ops` supplied
        are applied after the approximate year is inferred. `dod` is
        adjusted based on the inferred approximate year of care.

        """
        table = self.get_table(PATIENTS)

        # Process and include patient's anchor year.
        table = select(
            table,
            (
                func.substr(get_column(table, "anchor_year_group"), 1, 4).cast(Integer)
            ).label("anchor_year_group_start"),
            (
                func.substr(get_column(table, "anchor_year_group"), 8, 12).cast(Integer)
            ).label("anchor_year_group_end"),
        ).subquery()

        # Select the middle of the anchor year group as the anchor year
        table = select(
            table,
            (
                get_column(table, "anchor_year_group_start")
                + (
                    get_column(table, "anchor_year_group_end")
                    - get_column(table, "anchor_year_group_start")
                )
                / 2
            ).label("anchor_year_group_middle"),
        ).subquery()

        table = select(
            table,
            (
                get_column(table, "anchor_year_group_middle")
                - get_column(table, "anchor_year")
            ).label("anchor_year_difference"),
        ).subquery()

        # Shift relevant columns by anchor year difference
        table = qo.AddDeltaColumn("dod", years="anchor_year_difference")(table)
        table = qo.Drop(
            [
                "anchor_year_group_start",
                "anchor_year_group_end",
                "anchor_year_group_middle",
            ]
        )(table)

        return QueryInterface(self._db, table, join=join, ops=ops)

    def diagnoses(
        self,
        join: Optional[qo.JoinArgs] = None,
        ops: Optional[qo.Sequential] = None,
    ) -> QueryInterface:
        """Query MIMIC diagnosis data.

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
        table = self.get_table(DIAGNOSES)

        # Join with diagnoses dimension table.
        table = qo.Join(
            join_table=self.get_table(DIM_DIAGNOSES),
            on=["icd_code", "icd_version"],
            on_to_type=["str", "int"],
        )(table)

        return QueryInterface(self._db, table, join=join, ops=ops)

    def care_units(
        self, join: Optional[qo.JoinArgs] = None, ops: Optional[qo.Sequential] = None
    ) -> QueryInterfaceProcessed:
        """Get care unit table within a given set of encounters.

        Parameters
        ----------
        join: qo.JoinArgs, optional
            Join arguments.
        ops: qo.Sequential, optional
            Additional operations to apply to the query.

        Returns
        -------
        cyclops.query.interface.QueryInterfaceProcessed
            Constructed table, wrapped in an interface object.

        """
        table = self.get_table(TRANSFERS)
        return QueryInterfaceProcessed(
            self._db,
            table,
            process_fn=lambda x: process_mimic_care_units(x, specific=False),
            join=join,
            ops=ops,
        )

    def labevents(
        self, join: Optional[qo.JoinArgs] = None, ops: Optional[qo.Sequential] = None
    ) -> QueryInterface:
        """Query lab events from the hospital module.

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
        dim_items_table = self.get_table(DIM_LABITEMS)

        # Join with lab items dimension table.
        table = qo.Join(
            join_table=dim_items_table,
            on=["itemid"],
        )(table)

        return QueryInterface(self._db, table, join=join, ops=ops)

    def chartevents(
        self, join: Optional[qo.JoinArgs] = None, ops: Optional[qo.Sequential] = None
    ) -> QueryInterface:
        """Query ICU chart events from the ICU module.

        Parameters
        ----------
        join: qo.JoinArgs, optional
            Join arguments.
        ops: qo.Sequential, optional
            Additional operations to apply to the query.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed table, wrapped in an interface object.

        """
        table = self.get_table(CHARTEVENTS)
        dim_items_table = self.get_table(DIM_ITEMS)

        # Join with items dimension table.
        table = qo.Join(
            dim_items_table,
            on="itemid",
        )(table)

        return QueryInterface(self._db, table, join=join, ops=ops)
