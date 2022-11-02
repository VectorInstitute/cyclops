"""Test ORM implementation."""

from cyclops.config import read_config
from cyclops.constants import GEMINI, MIMICIV
from cyclops.orm import Database


def test_db_instantiation():
    """Test instantiation of DB that implements ORM."""
    # Attempts to connect to GEMINI and MIMIC but just prints logger warning.
    _ = Database(read_config(GEMINI))
    _ = Database(read_config(MIMICIV))
