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


# Constants.
PATIENT = "patient"
ADMISSIONDX = "admissiondx"
DIAGNOSIS = "diagnosis"
HOSPITAL = "hospital"
LAB = "lab"
MEDICATION = "medication"
VITALPERIODIC = "vitalperiodic"
VITALAPERIODIC = "vitalaperiodic"
RESPIRATORYCARE = "respiratorycare"
RESPIRATORYCHARTING = "respiratorycharting"
INTAKEOUTPUT = "intakeoutput"
MICROLAB = "microlab"
TREATMENT = "treatment"
INFUSIONDRUG = "infusiondrug"

TABLE_MAP = {
    PATIENT: lambda db: db.eicu_crd.patient,
    ADMISSIONDX: lambda db: db.eicu_crd.admissiondx,
    DIAGNOSIS: lambda db: db.eicu_crd.diagnosis,
    HOSPITAL: lambda db: db.eicu_crd.hospital,
    LAB: lambda db: db.eicu_crd.lab,
    MEDICATION: lambda db: db.eicu_crd.medication,
    VITALPERIODIC: lambda db: db.eicu_crd.vitalperiodic,
    VITALAPERIODIC: lambda db: db.eicu_crd.vitalaperiodic,
    RESPIRATORYCARE: lambda db: db.eicu_crd.respiratorycare,
    RESPIRATORYCHARTING: lambda db: db.eicu_crd.respiratorycharting,
    INTAKEOUTPUT: lambda db: db.eicu_crd.intakeoutput,
    MICROLAB: lambda db: db.eicu_crd.microlab,
    TREATMENT: lambda db: db.eicu_crd.treatment,
    INFUSIONDRUG: lambda db: db.eicu_crd.infusiondrug,
}


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
        super().__init__(TABLE_MAP, **overrides)
