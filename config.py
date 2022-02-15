"""Configuration module."""

import os
import logging
import subprocess
import getpass
import argparse
import configargparse
import time
import json
from datetime import datetime
from typing import Optional, Dict
from dotenv import load_dotenv


# Load environment vars.
load_dotenv()
from cyclops.utils.log import setup_logging, LOG_FILE_PATH  # noqa: E402


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=LOG_FILE_PATH, print_level="INFO", logger=LOGGER)


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
    if not config_path:
        parser = configargparse.ArgumentParser(
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            default_config_files=["./configs/default/*.yaml"],
        )
    else:
        parser = configargparse.ArgumentParser(
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            default_config_files=[config_path],
        )
    parser.add("-c", "--config_path", is_config_file=True, help="config file path")

    # ************************Operation.************************
    parser.add("--extract", action="store_true", help="Run data extraction")
    parser.add("--train", action="store_true", help="Execute model training code")
    parser.add("--predict", action="store_true", help="Run prediction")
    parser.add("--analyze", action="store_true", help="Run analysis")

    # ************************Data Extraction.************************
    # Database connection parameters.
    parser.add(
        "--user",
        type=str,
        required=False,
        default=os.environ["USER"],
        help="Database username",
    )
    parser.add(
        "--password",
        default=os.environ["PGPASSWORD"],
        type=str,
        required=False,
        help="Database password",
    )
    parser.add("--port", type=int, help="DB server port")
    parser.add("--host", type=str, required=False, help="DB server hostname")
    parser.add(
        "--dbms", type=str, required=False, help="DB system", choices=["postgresql"]
    )
    parser.add("--database", type=str, required=False, help="database name")

    # Data source and destination parameters.
    parser.add("-w", action="store_true", help="Write extracted data to disk")
    parser.add("-r", action="store_true", help="Read from the database")
    parser.add(
        "--input",
        type=str,
        required=False,
        help="Data file to read from instead of database",
    )
    parser.add(
        "--output_folder",
        type=str,
        help="Which directory should we put the CSV results?",
    )
    parser.add(
        "--output_full_path",
        type=str,
        help="Where should we put the CSV results? Full path option.",
    )
    parser.add(
        "--stats_path",
        type=str,
        help="Where to store/load features mean/std for normalization.",
    )
    parser.add(
        "--sql_query_path",
        type=str,
        help="Path to .sql file with SQL query.",
    )

    # Data extraction parameters.
    parser.add(
        "--features",
        default=[],
        type=str,
        action="append",
        required=False,
        help="List of features for the model",
    )
    parser.add(
        "--target", type=str, required=False, help="Column we are trying to predict"
    )
    parser.add(
        "--pop_size",
        type=int,
        required=False,
        help="Total number of records to read from the database (0 - to read all)",
    )
    parser.add(
        "--filter_year",
        type=int,
        required=False,
        help="Select only records from before specified year",
    )

    # Specify 'from' and 'to' dates, only records with admit_date
    # in this range will be selected.
    parser.add(
        "--filter_date_from",
        type=str,
        default="",
        required=False,
        help="Format: yyyy-mm-dd. Select starting from this admit_date.\
                Used in conjunction with --filter_date_to",
    )
    parser.add(
        "--filter_date_to",
        type=str,
        default="",
        required=False,
        help="Format: yyyy-mm-dd. Select before this admit_date.\
                Used in conjunction with --filter_date_from",
    )

    # Train/test/val split parameters.
    parser.add(
        "--split_column",
        type=str,
        required=False,
        help="Column we are use to split data into train, test, val",
    )
    parser.add("--test_split", type=str, required=False, help="Test split values")
    parser.add("--val_split", type=str, required=False, help="Val split values")
    parser.add(
        "--train_split",
        default=[],
        type=str,
        action="append",
        required=False,
        help="Train split values (if not set, all excdept test/val values)",
    )

    # ************************Model Training and Prediction.************************
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--model_path", type=str, default="./model.pt")

    # Data-loading config.
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--shuffle", action="store_true")

    # Used mostly for fake data, can take it out.
    parser.add_argument("--data_dim", type=int, default=24)
    parser.add_argument("--data_len", type=int, default=10000)

    # Training config.
    parser.add_argument("--lr", type=float, default=3e-4)

    # Prediction config.
    parser.add("--threshold", type=float, default=0.5)

    # Prediction output path.
    parser.add("--result_output", type=str, default="./result.csv")

    # ************************Analysis************************
    parser.add_argument(
        "--type", type=str, default="dataset", help="Type of report to generate"
    )

    # Data-specific parameters.
    parser.add(
        "--slice",
        type=str,
        required=False,
        help="What column to use to slice data for analysis?",
    )
    parser.add(
        "--data_ref",
        default=[],
        type=int,
        action="append",
        required=False,
        help="List of slices to take as reference data",
    )
    parser.add(
        "--data_eval",
        default=[],
        type=int,
        action="append",
        required=False,
        help="List of slices to evaluate on",
    )
    parser.add(
        "--numerical_features",
        default=[],
        type=str,
        action="append",
        required=False,
        help="List of numerical features (for analysis)",
    )
    parser.add(
        "--categorical_features",
        default=[],
        type=str,
        action="append",
        required=False,
        help="List of categorical features (for analysis)",
    )
    parser.add(
        "--report_path",
        type=str,
        required=False,
        help="Directory where to store html report?",
    )
    parser.add(
        "--report_full_path",
        default="",
        type=str,
        required=False,
        help="Full path for the report (filename is generated if not provided)",
    )
    parser.add(
        "-html",
        action="store_true",
        help="Produce HTML report (otherwise save json report)",
    )
    parser.add(
        "-target_num",
        action="store_true",
        required=False,
        help="Is target numerical (as opposed to categorical)",
    )
    parser.add(
        "--prediction_col",
        default="prediction",
        type=str,
        required=False,
        help="Name of the prediction column",
    )

    # Model performance parameters.
    parser.add(
        "--reference",
        type=str,
        required=False,
        help="Filename of features/prediction to use as reference",
    )
    parser.add(
        "--test",
        type=str,
        required=False,
        help="Filename of features/prediction to use as test\
                (for model drift evaluation)",
    )

    args, _ = parser.parse_known_args()

    args.commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("ascii")
    )

    if args.input is None and args.password is None:
        args.password = getpass.getpass(prompt="Database password: ", stream=None)

    if len(args.filter_date_from) and len(args.filter_date_to):
        args.filter_date_from = datetime.strptime(args.filter_date_from, "%Y%m%d")
        args.filter_date_to = datetime.strptime(args.filter_date_to, "%Y%m%d")

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


def write_config(config: argparse.Namespace):
    """Save configuration to file.

    Parameters
    ----------
    config: argparse.Namespace
        Configuration stored in object.
    """
    t = time.localtime()
    date = time.strftime("%Y-%b-%d_%H-%M-%S", t)

    LOGGER.info(f"Writing configuration with timestamp {date}!")
    config_to_save = config_to_dict(config)
    LOGGER.info(config_to_save)

    os.makedirs(config.output_folder, exist_ok=True)
    with open(os.path.join(config.output_folder, f"config_{date}.json"), "w") as fp:
        fp.write(json.dumps(config_to_save, indent=4))


if __name__ == "__main__":
    params = read_config()
    write_config(params)
