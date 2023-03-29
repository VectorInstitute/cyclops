"""The ``query`` API provides classes to query EHR databases."""


try:
    import sqlalchemy
except ImportError:
    raise ImportError(
        "CyclOps is not installed with query API support! Please install using 'pip install cyclops[query]'."  # noqa: E501 pylint: disable=line-too-long
    ) from None


from cyclops.query.eicu import EICUQuerier
from cyclops.query.gemini import GEMINIQuerier
from cyclops.query.mimiciii import MIMICIIIQuerier
from cyclops.query.mimiciv import MIMICIVQuerier
from cyclops.query.omop import OMOPQuerier
