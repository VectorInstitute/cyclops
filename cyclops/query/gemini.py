"""GEMINI query API."""

import logging
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.sql.expression import union_all
from sqlalchemy.sql.selectable import Subquery

from cyclops.process.column_names import (
    ADMIT_TIMESTAMP,
    CARE_UNIT,
    DIAGNOSIS_CODE,
    DISCHARGE_TIMESTAMP,
    ENCOUNTER_ID,
    ER_ADMIT_TIMESTAMP,
    ER_DISCHARGE_TIMESTAMP,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
    EVENT_VALUE_UNIT,
    HOSPITAL_ID,
    LENGTH_OF_STAY_IN_ER,
    SCU_ADMIT_TIMESTAMP,
    SCU_DISCHARGE_TIMESTAMP,
    SEX,
    SUBJECT_ID,
)
from cyclops.process.constants import EMPTY_STRING
from cyclops.query import process as qp
from cyclops.query.base import DatasetQuerier
from cyclops.query.interface import QueryInterface
from cyclops.query.util import (
    TableTypes,
    assert_table_has_columns,
    table_params_to_type,
)
from cyclops.utils.common import append_if_missing
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
TABLE_MAP = {
    IP_ADMIN: lambda db: db.public.ip_administrative,
    ER_ADMIN: lambda db: db.public.er_administrative,
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
COLUMN_MAP = {
    "genc_id": ENCOUNTER_ID,
    "admit_date_time": ADMIT_TIMESTAMP,
    "discharge_date_time": DISCHARGE_TIMESTAMP,
    "gender": SEX,
    "result_value": EVENT_VALUE,
    "result_unit": EVENT_VALUE_UNIT,
    "lab_test_name_mapped": EVENT_NAME,
    "sample_collection_date_time": EVENT_TIMESTAMP,
    "measurement_mapped": EVENT_NAME,
    "measurement_value": EVENT_VALUE,
    "measure_date_time": EVENT_TIMESTAMP,
    "left_er_date_time": ER_DISCHARGE_TIMESTAMP,
    "duration_er_stay_derived": LENGTH_OF_STAY_IN_ER,
    "er_admit_timestamp": ER_ADMIT_TIMESTAMP,
    "triage_date_time": ER_ADMIT_TIMESTAMP,
    "er_discharge_timestamp": ER_DISCHARGE_TIMESTAMP,
    "scu_admit_date_time": SCU_ADMIT_TIMESTAMP,
    "scu_discharge_date_time": SCU_DISCHARGE_TIMESTAMP,
    "hospital_id": HOSPITAL_ID,
    "diagnosis_code": DIAGNOSIS_CODE,
    "patient_id_hashed": SUBJECT_ID,
}
EVENT_CATEGORIES = [LAB, VITALS]


class GEMINIQuerier(DatasetQuerier):
    """GEMINI dataset querier."""

    def __init__(self, config_overrides: Optional[List] = None):
        """Initialize.

        Parameters
        ----------
        config_overrides: list, optional
            List of override configuration parameters.

        """
        if not config_overrides:
            config_overrides = []
        super().__init__(TABLE_MAP, COLUMN_MAP, config_overrides)

    def er_admin(self, **process_kwargs) -> QueryInterface:
        """Query emergency room administrative data.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed table, wrapped in an interface object.

        Other Parameters
        ----------------
        triage_level: int or list of int
            Restrict to certain triage levels.
        before_date: datetime.datetime or str
            Get data before some date.
            If a string, provide in YYYY-MM-DD format.
        after_date: datetime.datetime or str
            Get data after some date.
            If a string, provide in YYYY-MM-DD format.
        years: int or list of int, optional
            Get data by year.
        months: int or list of int, optional
            Get data by month.
        limit: int, optional
            Limit the number of rows returned.

        """
        table = self.get_table(ER_ADMIN)

        # Process optional operations
        operations: List[tuple] = [
            (qp.ConditionBeforeDate, [ER_ADMIT_TIMESTAMP, qp.QAP("before_date")], {}),
            (qp.ConditionAfterDate, [ER_ADMIT_TIMESTAMP, qp.QAP("after_date")], {}),
            (qp.ConditionInYears, [ER_ADMIT_TIMESTAMP, qp.QAP("years")], {}),
            (qp.ConditionInMonths, [ER_ADMIT_TIMESTAMP, qp.QAP("months")], {}),
            (
                qp.ConditionIn,
                ["triage_level", qp.QAP("triage_level")],
                {"to_str": True},
            ),
            (qp.Limit, [qp.QAP("limit")], {}),
        ]
        table = qp.process_operations(table, operations, process_kwargs)

        return QueryInterface(self._db, table)

    @table_params_to_type(Subquery)
    @assert_table_has_columns(er_admin_table=ENCOUNTER_ID)
    def patient_encounters(
        self,
        er_admin_table: Optional[TableTypes] = None,
        drop_null_subject_ids=True,
        **process_kwargs,
    ) -> QueryInterface:
        """Query GEMINI patient encounters.

        Parameters
        ----------
        er_admin_table: Subquery, optional
            Gather Emergency Room data recorded for the particular encounter.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed query, wrapped in an interface object.

        Other Parameters
        ----------------
        sex: str or list of string, optional
            Specify patient sex (one or multiple).
        died: bool, optional
            Specify True to get patients who have died, and False for those who haven't.
        died_binarize_col: str, optional
            Binarize the died condition and save as a column with label
            died_binarize_col.
        before_date: datetime.datetime or str
            Get patients encounters before some date.
            If a string, provide in YYYY-MM-DD format.
        after_date: datetime.datetime or str
            Get patients encounters after some date.
            If a string, provide in YYYY-MM-DD format.
        hospitals: str or list of str, optional
            Get patient encounters by hospital sites.
        years: int or list of int, optional
            Get patient encounters by year.
        months: int or list of int, optional
            Get patient encounters by month.
        limit: int, optional
            Limit the number of rows returned.

        """
        table = self.get_table(IP_ADMIN)

        if drop_null_subject_ids:
            table = qp.DropNulls(SUBJECT_ID)(table)

        # Possibly cast string representations to timestamps
        table = qp.Cast([ADMIT_TIMESTAMP, DISCHARGE_TIMESTAMP], "timestamp")(table)

        # Get the discharge disposition code descriptions
        lookup_table = self.get_table(LOOKUP_IP_ADMIN)
        lookup_table = qp.ConditionEquals("variable", "discharge_disposition")(
            lookup_table
        )

        table = qp.Join(
            lookup_table,
            on=("discharge_disposition", "value"),
            on_to_type="int",
            join_table_cols="description",
            isouter=True,
        )(table)
        table = qp.Rename({"description": "discharge_description"})(table)
        table = qp.Drop("value")(table)

        # Join on ER data only if specified
        if er_admin_table is not None:
            table = qp.Join(er_admin_table, on=ENCOUNTER_ID)(table)

        # Process optional operations
        if "died" not in process_kwargs and "died_binarize_col" in process_kwargs:
            process_kwargs["died"] = True

        operations: List[tuple] = [
            (qp.ConditionBeforeDate, ["admit_timestamp", qp.QAP("before_date")], {}),
            (qp.ConditionAfterDate, ["admit_timestamp", qp.QAP("after_date")], {}),
            (qp.ConditionInYears, ["admit_timestamp", qp.QAP("years")], {}),
            (qp.ConditionInMonths, ["admit_timestamp", qp.QAP("months")], {}),
            (qp.ConditionIn, [HOSPITAL_ID, qp.QAP("hospitals")], {"to_str": True}),
            (qp.ConditionIn, [SEX, qp.QAP("sex")], {"to_str": True}),
            (
                qp.ConditionEquals,
                ["discharge_description", "Died"],
                {
                    "not_": qp.QAP("died", transform_fn=lambda x: not x),
                    "binarize_col": qp.QAP("died_binarize_col", required=False),
                },
            ),
            (qp.Limit, [qp.QAP("limit")], {}),
        ]

        table = qp.process_operations(table, operations, process_kwargs)

        return QueryInterface(self._db, table)

    def diagnoses(self, **process_kwargs) -> QueryInterface:
        """Query diagnosis data.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed table, wrapped in an interface object.

        Other Parameters
        ----------------
        diagnosis_codes: str or list of str, optional
            Get only the specified ICD codes.
        diagnosis_types: list of str, optional
            Include only those diagnoses that are of certain type.
        limit: int, optional
            Limit the number of rows returned.

        Warnings
        --------
        Setting the ``include_description`` parameter would join diagnosis types
        with descriptions and if the diagnosis type is None, then those rows would
        be dropped.

        """
        table = self.get_table(DIAGNOSIS)

        lookup_table = self.get_table(LOOKUP_DIAGNOSIS)
        lookup_table = qp.ConditionEquals("variable", "diagnosis_type")(lookup_table)
        table = qp.Join(
            lookup_table,
            on=("diagnosis_type", "value"),
            join_table_cols="description",
            isouter=True,
        )(table)
        table = qp.Drop("value")(table)
        table = qp.Rename({"description": "diagnosis_type_description"})(table)
        table = qp.ReorderAfter("diagnosis_type_description", "diagnosis_type")(table)

        # Trim whitespace from ICD codes.
        table = qp.Trim(DIAGNOSIS_CODE)(table)

        # Process optional operations
        operations: List[tuple] = [
            (
                qp.ConditionIn,
                [DIAGNOSIS_CODE, qp.QAP("diagnosis_codes")],
                {"to_str": True},
            ),
            (
                qp.ConditionIn,
                ["diagnosis_type", qp.QAP("diagnosis_types")],
                {"to_str": True},
            ),
            (qp.Limit, [qp.QAP("limit")], {}),
        ]
        table = qp.process_operations(table, operations, process_kwargs)

        return QueryInterface(self._db, table)

    @table_params_to_type(Subquery)
    @assert_table_has_columns(
        diagnoses_table=[ENCOUNTER_ID, DIAGNOSIS_CODE],
        patient_encounters_table=[ENCOUNTER_ID, SUBJECT_ID],
    )
    def patient_diagnoses(
        self,
        diagnoses_table: Optional[TableTypes] = None,
        patient_encounters_table: Optional[TableTypes] = None,
        **process_kwargs,
    ) -> QueryInterface:
        """Query diagnosis data.

        Parameters
        ----------
        diagnoses_table: cyclops.query.util.TableTypes, optional
            Diagnoses table used to join.
        patient_encounters_table: cyclops.query.util.TableTypes, optional
            Patient encounters table used to join.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed table, wrapped in an interface object.

        Other Parameters
        ----------------
        limit: int, optional
            Limit the number of rows returned.

        """
        # Get diagnoses
        if diagnoses_table is None:
            diagnoses_table = self.diagnoses().query

        # Get patient encounters
        if patient_encounters_table is None:
            patient_encounters_table = self.patient_encounters().query

        # Join on patient encounters
        table = qp.Join(diagnoses_table, on=ENCOUNTER_ID, isouter=True)(
            patient_encounters_table
        )

        # Process optional operations
        operations: List[tuple] = [(qp.Limit, [qp.QAP("limit")], {})]
        table = qp.process_operations(table, operations, process_kwargs)

        return QueryInterface(self._db, table)

    def room_transfers(self, **process_kwargs) -> QueryInterface:
        """Query room transfer data.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed table, wrapped in an interface object.

        Other Parameters
        ----------------
        limit: int, optional
            Limit the number of rows returned.

        """
        table = self.get_table(ROOM_TRANSFER)

        # Join with lookup to get transfer description.
        lookup_table = self.get_table(LOOKUP_ROOM_TRANSFER)
        lookup_table = qp.ConditionEquals("variable", "medical_service")(lookup_table)

        table = qp.Join(
            lookup_table,
            on=("medical_service", "value"),
            join_table_cols="description",
            isouter=True,
        )(table)
        table = qp.Rename({"description": "transfer_description"})(table)

        # Process optional operations
        operations: List[tuple] = [(qp.Limit, [qp.QAP("limit")], {})]
        table = qp.process_operations(table, operations, process_kwargs)

        return QueryInterface(self._db, table)

    @table_params_to_type(Subquery)
    @assert_table_has_columns(patient_encounters_table=[ENCOUNTER_ID, SUBJECT_ID])
    def care_units(
        self,
        patient_encounters_table: Optional[TableTypes] = None,
        **process_kwargs,
    ) -> QueryInterface:
        """Query care unit data.

        Parameters
        ----------
        patient_encounters_table: cyclops.query.util.TableTypes, optional
            Patient encounters table used to join.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed table, wrapped in an interface object.

        Other Parameters
        ----------------
        limit: int, optional
            Limit the number of rows returned.

        """
        filter_care_unit_cols = qp.FilterColumns(
            [
                ENCOUNTER_ID,
                "admit",
                "discharge",
                CARE_UNIT,
            ]
        )

        # In-patient table.
        ip_table = self.get_table(IP_ADMIN)
        ip_table = qp.Rename(
            {
                ADMIT_TIMESTAMP: "admit",
                DISCHARGE_TIMESTAMP: "discharge",
            }
        )(ip_table)
        ip_table = qp.Literal("IP", CARE_UNIT)(ip_table)
        ip_table = filter_care_unit_cols(ip_table)

        # Special care unit table.
        scu_table = self.get_table(IP_SCU)
        scu_table = qp.Rename(
            {
                SCU_ADMIT_TIMESTAMP: "admit",
                SCU_DISCHARGE_TIMESTAMP: "discharge",
            }
        )(scu_table)
        scu_table = qp.Literal("SCU", CARE_UNIT)(scu_table)
        scu_table = filter_care_unit_cols(scu_table)

        # Emergency room/department table.
        er_table = self.er_admin().query
        er_table = qp.Rename(
            {
                ER_ADMIT_TIMESTAMP: "admit",
                ER_DISCHARGE_TIMESTAMP: "discharge",
            }
        )(er_table)
        er_table = qp.Literal("ER", CARE_UNIT)(er_table)
        er_table = filter_care_unit_cols(er_table)

        # Room transfer table.
        rt_table = self.room_transfers().query
        rt_table = qp.Rename(
            {
                "checkin_date_time": "admit",
                "checkout_date_time": "discharge",
            }
        )(rt_table)
        rt_table = qp.Rename({"transfer_description": CARE_UNIT})(rt_table)
        rt_table = filter_care_unit_cols(rt_table)

        # Combine.
        table = union_all(
            select(er_table),
            select(scu_table),
            select(ip_table),
            select(rt_table),
        ).subquery()

        if patient_encounters_table is not None:
            table = qp.Join(patient_encounters_table, on=ENCOUNTER_ID)(table)

        # Process optional operations
        operations: List[tuple] = [(qp.Limit, [qp.QAP("limit")], {})]
        table = qp.process_operations(table, operations, process_kwargs)

        return QueryInterface(self._db, table)

    @table_params_to_type(Subquery)
    @assert_table_has_columns(patient_encounters_table=[ENCOUNTER_ID, SUBJECT_ID])
    def events(
        self,
        event_category: str,
        patient_encounters_table: Optional[TableTypes] = None,
        drop_null_event_names: bool = True,
        drop_null_event_values: bool = False,
        **process_kwargs,
    ) -> QueryInterface:
        """Query events.

        Parameters
        ----------
        event_category : str or list of str
            Specify event category, e.g., lab, vitals, intervention, etc.
        patient_encounters_table: cyclops.query.util.TableTypes, optional
            Patient encounters table used to join.
        drop_null_event_names: bool, default = True
            Whether to drop rows with null or empty event names.
        drop_null_event_values: bool, default = False
            Whether to drop rows with null event values.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed table, wrapped in an interface object.


        Other Parameters
        ----------------
        event_names: str or list of str, optional
            Get only certain event names.
        event_name_substring: str, optional
            Get only event names with some substring(s).
        limit: int, optional
            Limit the number of rows returned.

        """
        if event_category not in EVENT_CATEGORIES:
            raise ValueError(
                f"""Invalid event category specified.
                Must be in {", ".join(EVENT_CATEGORIES)}"""
            )

        table = self.get_table(event_category)

        # Remove rows with null events/events with no recorded name
        if drop_null_event_names:
            table = qp.DropNulls(EVENT_NAME)(table)
            table = qp.ConditionEquals(
                EVENT_NAME, EMPTY_STRING, not_=True, to_str=True
            )(table)

        # Remove rows with null event values
        if drop_null_event_values:
            table = qp.DropNulls(EVENT_VALUE)(table)

        # Process optional operations
        operations: List[tuple] = [
            (qp.ConditionIn, [EVENT_NAME, qp.QAP("event_names")], {"to_str": True}),
            (qp.ConditionBeforeDate, [EVENT_TIMESTAMP, qp.QAP("before_date")], {}),
            (qp.ConditionAfterDate, [EVENT_TIMESTAMP, qp.QAP("after_date")], {}),
            (qp.ConditionInYears, [EVENT_TIMESTAMP, qp.QAP("years")], {}),
            (qp.ConditionInMonths, [EVENT_TIMESTAMP, qp.QAP("months")], {}),
            (
                qp.ConditionSubstring,
                [EVENT_NAME, qp.QAP("event_name_substring")],
                {"to_str": True},
            ),
            (qp.Limit, [qp.QAP("limit")], {}),
        ]

        table = qp.process_operations(table, operations, process_kwargs)

        # Join on patient encounters
        if patient_encounters_table is not None:
            table = qp.Join(patient_encounters_table, on=ENCOUNTER_ID)(table)

        return QueryInterface(self._db, table)

    def blood_transfusions(self, **process_kwargs) -> QueryInterface:
        """Query blood transfusion data.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed table, wrapped in an interface object.

        Other Parameters
        ----------------
        rbc_mapped: bool, optional
            Whether the patient was transfused with Red Blood Cells.
        rbc_mapped_binarize_col: str, optional
            Binarize the rbc_mapped condition and save as a column with
            label rbc_mapped_binarize_col.
        blood_product_raw_substring: str, optional
            Get only blood_product_raw rows with some substring(s).
        blood_product_raw_names
            Get only specified blood_product_raw rows.
        before_date: datetime.datetime or str
            Get tranfusions before some date.
            If a string, provide in YYYY-MM-DD format.
        after_date: datetime.datetime or str
            Get tranfusions after some date.
            If a string, provide in YYYY-MM-DD format.
        years: int or list of int, optional
            Get tranfusions by year.
        months: int or list of int, optional
            Get tranfusions by month.
        limit: int, optional
            Limit the number of rows returned.

        """
        table = self.get_table(BLOOD_TRANSFUSION)

        table = qp.Cast("issue_date_time", "timestamp")(table)

        operations: List[tuple] = [
            (qp.ConditionBeforeDate, ["issue_date_time", qp.QAP("before_date")], {}),
            (qp.ConditionAfterDate, ["issue_date_time", qp.QAP("after_date")], {}),
            (qp.ConditionInYears, ["issue_date_time", qp.QAP("years")], {}),
            (qp.ConditionInMonths, ["issue_date_time", qp.QAP("months")], {}),
            (qp.ConditionEquals, ["rbc_mapped", qp.QAP("rbc_mapped")], {}),
            (
                qp.ConditionSubstring,
                ["blood_product_raw", qp.QAP("blood_product_raw_substring")],
                {},
            ),
            (
                qp.ConditionIn,
                ["blood_product_raw", qp.QAP("blood_product_raw_names")],
                {"to_str": True},
            ),
            (qp.Limit, [qp.QAP("limit")], {}),
        ]
        table = qp.process_operations(table, operations, process_kwargs)

        return QueryInterface(self._db, table)

    def interventions(self, **process_kwargs) -> QueryInterface:
        """Query interventions data.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed table, wrapped in an interface object.

        Other Parameters
        ----------------
        limit: int, optional
            Limit the number of rows returned.
        years: int or list of int, optional
            Get tests by year.

        """
        table = self.get_table(INTERVENTION)

        operations: List[tuple] = [
            (
                qp.ConditionInYears,
                ["intervention_episode_start_date", qp.QAP("years")],
                {},
            ),
            (qp.Limit, [qp.QAP("limit")], {}),
        ]
        table = qp.process_operations(table, operations, process_kwargs)

        return QueryInterface(self._db, table)

    def imaging(self, **process_kwargs) -> QueryInterface:
        """Query imaging data.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed table, wrapped in an interface object.

        Other Parameters
        ----------------
        test_descriptions: str or list of str
            Get only certain tests with the specified descriptions.
        raw_test_names: str or list of str
            Get only certain raw test names.
        before_date: datetime.datetime or str
            Get tests before some date.
            If a string, provide in YYYY-MM-DD format.
        after_date: datetime.datetime or str
            Get tests after some date.
            If a string, provide in YYYY-MM-DD format.
        years: int or list of int, optional
            Get tests by year.
        months: int or list of int, optional
            Get tests by month.
        limit: int, optional
            Limit the number of rows returned.

        """
        table = self.get_table(IMAGING)

        # Get imaging test description
        lookup_table = self.get_table(LOOKUP_IMAGING)
        lookup_table = qp.ConditionEquals("variable", "imaging_test_name_mapped")(
            lookup_table
        )

        table = qp.Join(
            lookup_table,
            on=("imaging_test_name_mapped", "value"),
            on_to_type="str",
            join_table_cols="description",
        )(table)
        table = qp.Drop("value")(table)
        table = qp.Rename({"description": "imaging_test_description"})(table)
        table = qp.ReorderAfter("imaging_test_description", "imaging_test_name_mapped")(
            table
        )

        operations: List[tuple] = [
            (
                qp.ConditionBeforeDate,
                ["performed_date_time", qp.QAP("before_date")],
                {},
            ),
            (qp.ConditionAfterDate, ["performed_date_time", qp.QAP("after_date")], {}),
            (qp.ConditionInYears, ["performed_date_time", qp.QAP("years")], {}),
            (qp.ConditionInMonths, ["performed_date_time", qp.QAP("months")], {}),
            (
                qp.ConditionIn,
                ["imaging_test_description", qp.QAP("test_descriptions")],
                {},
            ),
            (
                qp.ConditionIn,
                ["imaging_test_name_raw", qp.QAP("raw_test_names")],
                {"to_str": True},
            ),
            (qp.Limit, [qp.QAP("limit")], {}),
        ]
        table = qp.process_operations(table, operations, process_kwargs)

        return QueryInterface(self._db, table)

    def derived_variables(self, **process_kwargs) -> QueryInterface:
        """Query derived variable data.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed table, wrapped in an interface object.

        Other Parameters
        ----------------
        variables: str or list of str
            Variable columns to keep.
        limit: int, optional
            Limit the number of rows returned.

        """
        table = self.get_table(DERIVED_VARIABLES)

        operations: List[tuple] = [
            (
                qp.FilterColumns,
                [
                    qp.QAP(
                        "variables",
                        transform_fn=lambda x: append_if_missing(
                            x, ENCOUNTER_ID, to_start=True
                        ),
                    )
                ],
                {},
            ),
            (qp.Limit, [qp.QAP("limit")], {}),
        ]
        table = qp.process_operations(table, operations, process_kwargs)

        return QueryInterface(self._db, table)

    def pharmacy(self, **process_kwargs) -> QueryInterface:
        """Query pharmacy data.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed table, wrapped in an interface object.

        Other Parameters
        ----------------
        limit: int, optional
            Limit the number of rows returned.

        """
        table = self.get_table(PHARMACY)

        operations: List[tuple] = [
            (qp.Limit, [qp.QAP("limit")], {}),
        ]
        table = qp.process_operations(table, operations, process_kwargs)

        return QueryInterface(self._db, table)
