"""Configuration module."""

import os
import glob
import time
import logging
import json
from typing import Optional, Dict
import subprocess
import argparse
import configargparse
from dotenv import load_dotenv

from codebase_ops import get_log_file_path, PROJECT_ROOT
from cyclops.utils.log import setup_logging
from cyclops.constants import GEMINI, MIMIC


# Load environment vars.
load_dotenv()


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


DEFAULT_CONFIG_PATHS_MAP = {
    GEMINI: os.path.join(PROJECT_ROOT, "configs/gemini/*.yaml"),
    MIMIC: os.path.join(PROJECT_ROOT, "configs/mimic/*.yaml"),
}


def _check_if_config_files_exist(config_path):
    """Check if all four config files exist in mentioned path."""
    config_file_paths = glob.glob(config_path)
    config_file_names = [
        os.path.splitext(os.path.basename(file_path))[0]
        for file_path in config_file_paths
    ]
    if sorted(config_file_names) != ["data", "drift", "model", "workflow"]:
        return False
    return True


def _create_parser(
    config_path: str = DEFAULT_CONFIG_PATHS_MAP[GEMINI],
) -> configargparse.ArgumentParser:
    """Create argument parser.

    Parameters
    ----------
    config_path: str
        Path to config files.

    Returns
    -------
    configargparse.ArgumentParser
        ArgumentParser object to help input arguments, that override parameters
        from the configuration files.

    """
    if DEFAULT_CONFIG_PATHS_MAP.get(config_path):
        config_path = DEFAULT_CONFIG_PATHS_MAP[config_path]
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, "*.yaml")

    if not _check_if_config_files_exist(config_path):
        raise AssertionError(
            "Specified configuration folder does not have all configuration files!"
        )

    return configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        default_config_files=[config_path],
    )


def _add_sub_task_args(parser: configargparse.ArgumentParser) -> None:
    """Add sub-task toggle flags to argument parser."""
    parser.add("--extract", action="store_true", help="Run data extraction")
    parser.add("--train", action="store_true", help="Train baseline models")
    parser.add("--predict", action="store_true", help="Run prediction")
    parser.add("--drift", action="store_true", help="Run drift experiment")


def _add_data_query_args(parser: configargparse.ArgumentParser) -> None:
    """Add data querying parameters to argument parser."""
    parser.add(
        "--user",
        type=str,
        required=False,
        help="Username for querying database.",
    )
    parser.add(
        "--password",
        default=os.environ.get("PGPASSWORD", None),
        type=str,
        required=False,
        help="Database password.",
    )
    parser.add("--port", type=int, help="DB server port.")
    parser.add("--host", type=str, required=False, help="DB server hostname.")
    parser.add(
        "--dbms", type=str, required=False, help="DB system.", choices=["postgresql"]
    )
    parser.add("--database", type=str, required=False, help="Name of the database.")
    parser.add(
        "--years",
        type=str,
        default=[],
        action="append",
        required=False,
        help="Extract data filtered from specified year(s).",
    )
    parser.add(
        "--hospitals",
        type=str,
        default=[],
        action="append",
        required=False,
        help="Extract data filtered from specified hospital(s) sites.",
    )
    parser.add(
        "--from_date",
        type=str,
        default="",
        required=False,
        help="""Format: yyyy-mm-dd. Filter to include patient encounters starting
        from this date.""",
    )
    parser.add(
        "--to_date",
        type=str,
        default="",
        required=False,
        help="""Format: yyyy-mm-dd. Filter to include patient encounters before this
        date.""",
    )
    parser.add(
        "--output_folder",
        type=str,
        default="_extract",
        help="Output folder to store extracted data.",
    )


def _add_data_process_args(parser: configargparse.ArgumentParser) -> None:
    """Add data processing parameters to argument parser."""
    parser.add(
        "--aggregation_strategy",
        type=str,
        default="static",
        required=False,
        help="""Aggregation strategy. If 'static', then patient events from time window
        are collapsed to statistics per encounter. If 'temporal',
        then patient events are aggregrated into smaller time windows specified
        by aggregation_window. 'static' processing is suitable for time-invariant
        models, and 'temporal' for time-series or recurrent models.""",
    )
    parser.add(
        "--aggregation_range",
        type=int,
        default=168,
        required=False,
        help="""No. of hours after admission to consider for aggregating patient
        events data.""",
    )
    parser.add(
        "--aggregation_window",
        type=int,
        default=24,
        required=False,
        help="No. of hours (window) to consider for aggregating patient events data.",
    )
    parser.add(
        "--imputation_strategy",
        type=str,
        default="mean",
        required=False,
        choices=[None, "mean"],
        help="""Imputation strategy to apply for missing values.""",
    )


def _get_commit_id() -> str:
    """Get the commit id of HEAD."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("ascii")


def read_config(config_path: Optional[str] = None) -> argparse.Namespace:
    """Read configuration.

    Parameters
    ----------
    config_path: str, optional
        Path to config files.

    Returns
    -------
    args: argparse.Namespace
        Configuration stored in object.

    """
    if config_path:
        parser = _create_parser(config_path)
    else:
        parser = _create_parser()

    _add_sub_task_args(parser)
    _add_data_query_args(parser)
    _add_data_process_args(parser)

    args, _ = parser.parse_known_args()

    args.commit = _get_commit_id()
    if not args.user:
        args.user = os.environ["USER"]
    if args.password is None:
        LOGGER.warning(
            "DB password is not set! Add it to config, or PGPASSWORD env variable!"
        )

    return args


def config_to_dict(config: argparse.Namespace) -> Dict:
    """Create dict out of config, removing password.

    Parameters
    ----------
    config: argparse.Namespace
        Configuration stored in object.

    Returns
    -------
    dict
        dict with configuration parameters.

    """
    return {k: v for k, v in vars(config).items() if k != "password"}


def write_config(config: argparse.Namespace) -> str:
    """Save configuration parameters to file (for bookkeeping/tracking experiments).

    Parameters
    ----------
    config: argparse.Namespace
        Configuration stored in object.

    Returns
    -------
    str
        Path to saved config parameters in json format.

    """
    date = time.strftime("%Y-%b-%d_%H-%M-%S", time.localtime())

    LOGGER.info("Writing configuration with timestamp %s!", date)
    config_to_save = config_to_dict(config)
    LOGGER.info(config_to_save)

    os.makedirs(config.output_folder, exist_ok=True)
    file_path = os.path.join(config.output_folder, f"config_{date}.json")
    with open(file_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(config_to_save, indent=4))
    return file_path
