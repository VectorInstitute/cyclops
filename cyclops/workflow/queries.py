"""Queries to run in pipeline."""

from typing import Callable, Iterator, Optional

from cyclops.query import mimic

QUERY_CATELOG = {}


def register_query(query_fn: Callable, name: Optional[str] = None) -> None:
    """Register query function with dict.

    Parameters
    ----------
    query_fn: Callable
        Function that creates some queries to run.
    name: str, optional

    """
    if not name:
        name = query_fn.__name__
    if name not in QUERY_CATELOG:
        QUERY_CATELOG[name] = query_fn


def example_mimic_query() -> Iterator:
    """Showcase an example set of queries on mimic.

    Returns
    -------
    Iterator
        Iterator with queries to run.

    """
    # Query functions.
    patients = mimic.patients(from_date="2009-01-01", to_date="2009-01-06")
    patients_diagnoses = mimic.diagnoses(patients=patients)
    patients_labs = mimic.events(patients=patients, category="labs")
    patients_vitals = mimic.events(patients=patients, category="routine vital signs")
    patients_transfers = mimic.care_units(patients=patients)

    return zip(
        ["diagnoses", "labs", "vitals", "transfers"],
        [patients_diagnoses, patients_labs, patients_vitals, patients_transfers],
    )


# Register query creation functions.
register_query(example_mimic_query)
