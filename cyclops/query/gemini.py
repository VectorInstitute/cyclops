"""GEMINI query API."""

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.sql.expression import union_all

import cyclops.query.ops as qo
from cyclops.query.base import DatasetQuerier
from cyclops.query.interface import QueryInterface
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


# Constants.
IP_ADMIN = "ip_admin"
ER_ADMIN = "er_admin"
DIAGNOSIS = "diagnosis"
LAB = "lab"
VITALS = "vitals"
PHARMACY = "pharmacy"
INTERVENTION = "intervention"
LOOKUP_IP_ADMIN = "lookup_ip_admin"
LOOKUP_ER_ADMIN = "lookup_er_admin"
LOOKUP_DIAGNOSIS = "lookup_diagnosis"
LOOKUP_CCSR = "lookup_ccsr"
IP_SCU = "ip_scu"
LOOKUP_ROOM_TRANSFER = "lookup_room_transfer"
ROOM_TRANSFER = "room_transfer"
BLOOD_TRANSFUSION = "blood_transfusion"
IMAGING = "imaging"
LOOKUP_IMAGING = "lookup_imaging"
DERIVED_VARIABLES = "derived_variables"

# Custm column names.
CARE_UNIT = "care_unit"

# Table map.
TABLE_MAP = {
    ER_ADMIN: lambda db: db.public.er_administrative,
    IP_ADMIN: lambda db: db.public.ip_administrative,
    DIAGNOSIS: lambda db: db.public.diagnosis,
    LAB: lambda db: db.public.lab,
    VITALS: lambda db: db.public.vitals,
    PHARMACY: lambda db: db.public.pharmacy,
    INTERVENTION: lambda db: db.public.intervention,
    LOOKUP_IP_ADMIN: lambda db: db.public.lookup_ip_administrative,
    LOOKUP_ER_ADMIN: lambda db: db.public.lookup_er_administrative,
    LOOKUP_DIAGNOSIS: lambda db: db.public.lookup_diagnosis,
    LOOKUP_CCSR: lambda db: db.public.lookup_ccsr,
    IP_SCU: lambda db: db.public.ip_scu,
    ROOM_TRANSFER: lambda db: db.public.room_transfer,
    LOOKUP_ROOM_TRANSFER: lambda db: db.public.lookup_room_transfer,
    BLOOD_TRANSFUSION: lambda db: db.public.blood_transfusion,
    IMAGING: lambda db: db.public.imaging,
    LOOKUP_IMAGING: lambda db: db.public.lookup_imaging,
    DERIVED_VARIABLES: lambda db: db.public.derived_variables,
}


class GEMINIQuerier(DatasetQuerier):
    """GEMINI dataset querier."""

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

    def ip_admin(
        self,
        join: Optional[qo.JoinArgs] = None,
        ops: Optional[qo.Sequential] = None,
    ) -> QueryInterface:
        """Query GEMINI patient encounters.

        Parameters
        ----------
        join: qo.JoinArgs, optional
            Join arguments.
        ops: qo.Sequential, optional
            Additional operations to perform on the table.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed query, wrapped in an interface object.

        """
        table = self.get_table(IP_ADMIN)

        # Possibly cast string representations to timestamps
        table = qo.Cast(["admit_date_time", "discharge_date_time"], "timestamp")(table)

        # Get the discharge disposition code descriptions
        lookup_table = self.get_table(LOOKUP_IP_ADMIN)
        lookup_table = qo.ConditionEquals("variable", "discharge_disposition")(
            lookup_table
        )
        table = qo.Join(
            lookup_table,
            on=("discharge_disposition", "value"),
            on_to_type="int",
            join_table_cols="description",
            isouter=True,
        )(table)
        table = qo.Rename({"description": "discharge_description"})(table)
        table = qo.Drop("value")(table)

        return QueryInterface(self._db, table, join=join, ops=ops)

    def diagnoses(
        self, join: Optional[qo.JoinArgs] = None, ops: Optional[qo.Sequential] = None
    ) -> QueryInterface:
        """Query diagnosis data.

        Parameters
        ----------
        join: qo.JoinArgs, optional
            Join arguments.
        ops: qo.Sequential, optional
            Additional operations to perform on the table.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed table, wrapped in an interface object.

        """
        table = self.get_table(DIAGNOSIS)

        lookup_table = self.get_table(LOOKUP_DIAGNOSIS)
        lookup_table = qo.ConditionEquals("variable", "diagnosis_type")(lookup_table)
        table = qo.Join(
            lookup_table,
            on=("diagnosis_type", "value"),
            join_table_cols="description",
            isouter=True,
        )(table)
        table = qo.Drop("value")(table)
        table = qo.Rename({"description": "diagnosis_type_description"})(table)
        table = qo.ReorderAfter("diagnosis_type_description", "diagnosis_type")(table)

        # Trim whitespace from ICD codes.
        table = qo.Trim("diagnosis_code")(table)

        return QueryInterface(self._db, table, join=join, ops=ops)

    def room_transfer(
        self, join: Optional[qo.JoinArgs] = None, ops: Optional[qo.Sequential] = None
    ) -> QueryInterface:
        """Query room transfer data.

        Parameters
        ----------
        join: qo.JoinArgs, optional
            Join arguments.
        ops: qo.Sequential, optional
            Additional operations to perform on the table.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed table, wrapped in an interface object.

        """
        table = self.get_table(ROOM_TRANSFER)

        # Join with lookup to get transfer description.
        lookup_table = self.get_table(LOOKUP_ROOM_TRANSFER)
        lookup_table = qo.ConditionEquals("variable", "medical_service")(lookup_table)

        table = qo.Join(
            lookup_table,
            on=("medical_service", "value"),
            join_table_cols="description",
            isouter=True,
        )(table)
        table = qo.Rename({"description": "transfer_description"})(table)

        return QueryInterface(self._db, table, join=join, ops=ops)

    def care_units(
        self,
        join: Optional[qo.JoinArgs] = None,
        ops: Optional[qo.Sequential] = None,
    ) -> QueryInterface:
        """Query care unit data, fetches transfer info from multiple tables.

        Parameters
        ----------
        join: qo.JoinArgs, optional
            Join arguments.
        ops: qo.Sequential, optional
            Additional operations to perform on the table.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed table, wrapped in an interface object.

        """
        filter_care_unit_cols = qo.Keep(
            [
                "genc_id",
                "admit",
                "discharge",
                CARE_UNIT,
            ]
        )

        # In-patient table.
        ip_table = self.get_table(IP_ADMIN)
        ip_table = qo.Rename(
            {
                "admit_date_time": "admit",
                "discharge_date_time": "discharge",
            }
        )(ip_table)
        ip_table = qo.Literal("IP", CARE_UNIT)(ip_table)
        ip_table = filter_care_unit_cols(ip_table)

        # Special care unit table.
        scu_table = self.get_table(IP_SCU)
        scu_table = qo.Rename(
            {
                "scu_admit_date_time": "admit",
                "scu_discharge_date_time": "discharge",
            }
        )(scu_table)
        scu_table = qo.Literal("SCU", CARE_UNIT)(scu_table)
        scu_table = filter_care_unit_cols(scu_table)

        # Emergency room/department table.
        er_table = self.get_table(ER_ADMIN)
        er_table = qo.Rename(
            {
                "er_admit_timestamp": "admit",
                "er_discharge_timestamp": "discharge",
            }
        )(er_table)
        er_table = qo.Literal("ER", CARE_UNIT)(er_table)
        er_table = filter_care_unit_cols(er_table)

        # Room transfer table.
        rt_table = self.get_table(ROOM_TRANSFER)
        rt_table = qo.Rename(
            {
                "checkin_date_time": "admit",
                "checkout_date_time": "discharge",
            }
        )(rt_table)
        rt_table = qo.Rename({"transfer_description": CARE_UNIT})(rt_table)
        rt_table = filter_care_unit_cols(rt_table)

        # Combine.
        table = union_all(
            select(er_table),
            select(scu_table),
            select(ip_table),
            select(rt_table),
        ).subquery()

        return QueryInterface(self._db, table, join=join, ops=ops)

    def imaging(
        self, join: Optional[qo.JoinArgs] = None, ops: Optional[qo.Sequential] = None
    ) -> QueryInterface:
        """Query imaging reports data.

        Parameters
        ----------
        join: qo.JoinArgs, optional
            Join arguments.
        ops: qo.Sequential, optional
            Additional operations to perform on the table.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed table, wrapped in an interface object.

        """
        table = self.get_table(IMAGING)

        # Get imaging test description
        lookup_table = self.get_table(LOOKUP_IMAGING)
        lookup_table = qo.ConditionEquals("variable", "imaging_test_name_mapped")(
            lookup_table
        )

        table = qo.Join(
            lookup_table,
            on=("imaging_test_name_mapped", "value"),
            on_to_type="str",
            join_table_cols="description",
        )(table)
        table = qo.Drop("value")(table)
        table = qo.Rename({"description": "imaging_test_description"})(table)
        table = qo.ReorderAfter("imaging_test_description", "imaging_test_name_mapped")(
            table
        )

        return QueryInterface(self._db, table, join=join, ops=ops)
