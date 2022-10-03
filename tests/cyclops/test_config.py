"""Test configuration module."""

import json
import os
import shutil
import unittest

import pytest
import yaml

from codebase_ops import PROJECT_ROOT
from cyclops import config


def _save_to_yaml(to_save: dict, file_path: str) -> None:
    with open(file_path, "w", encoding="utf8") as file_handle:
        yaml.safe_dump(
            to_save,
            file_handle,
        )


class TestConfig(unittest.TestCase):
    """Test cases for config."""

    def setUp(self):
        """Set up dummy config and output directories."""
        self.dummy_config_dir = "/tmp/config"
        os.makedirs(self.dummy_config_dir, exist_ok=True)
        dummy_cfg = {"database": "test", "years": [2020, 2019]}
        _save_to_yaml(dummy_cfg, os.path.join(self.dummy_config_dir, "query.yaml"))
        _save_to_yaml({}, os.path.join(self.dummy_config_dir, "model.yaml"))
        _save_to_yaml({}, os.path.join(self.dummy_config_dir, "processor.yaml"))
        _save_to_yaml({}, os.path.join(self.dummy_config_dir, "workflow.yaml"))

    def tearDown(self):
        """Remove temp files after running tests."""
        shutil.rmtree(self.dummy_config_dir)

    def test_read_config(self):
        """Test reading configuration."""
        os.environ["PGPASSWORD"] = "test"
        cfg = config.read_config()
        assert cfg.password == "test"

        cfg = config.read_config(os.path.join(PROJECT_ROOT, "configs/mimiciv"))
        assert cfg.database == "mimiciv-2.0"

        cfg = config.read_config(os.path.join(PROJECT_ROOT, "configs/mimiciv/*.yaml"))
        assert cfg.database == "mimiciv-2.0"

        cfg = config.read_config("mimiciv")
        assert cfg.database == "mimiciv-2.0"

        cfg = config.read_config("gemini")
        assert cfg.host == "db.gemini-hpc.ca"

        cfg = config.read_config(self.dummy_config_dir)
        assert cfg.database == "test"
        assert cfg.years == ["2020", "2019"]

        if os.path.isfile(os.path.join(self.dummy_config_dir, "workflow.yaml")):
            os.remove(os.path.join(self.dummy_config_dir, "workflow.yaml"))

        with pytest.raises(AssertionError):
            _ = config.read_config(self.dummy_config_dir)
        _save_to_yaml({}, os.path.join(self.dummy_config_dir, "workflow.yaml"))

        os.environ.pop("PGPASSWORD", None)
        _ = config.read_config(self.dummy_config_dir)

    def test_config_to_dict(self):
        """Test config_to_dict fn."""
        os.environ["PGPASSWORD"] = "test"
        cfg = config.read_config(self.dummy_config_dir)
        cfg_dict = config.config_to_dict(cfg)
        assert isinstance(cfg_dict, dict)
        assert "password" not in cfg_dict

    def test_write_config(self):
        """Test write_config fn."""
        os.environ["PGPASSWORD"] = "test"
        cfg = config.read_config(self.dummy_config_dir)
        file_path = config.write_config(cfg)
        assert os.path.isfile(file_path)
        with open(file_path, "r", encoding="utf8") as file_handle:
            cfg_written = json.load(file_handle)
        assert cfg_written["database"] == "test"
        assert cfg_written["years"] == ["2020", "2019"]
