"""Test ORM implementation."""

import pytest
import socket

from cyclops.config import read_config
from cyclops.orm import Database
from cyclops.constants import GEMINI


def test_db_instantiation():
    """Test instantiation of DB that implements ORM."""
    # Attempts to connect to GEMINI and runs into error.
    with pytest.raises(socket.gaierror):
        _ = Database(read_config(GEMINI))
