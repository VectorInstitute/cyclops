"""Pipeline module that defines different tasks that can be executed in a workflow."""

import os

import config as config_parser
import luigi
from luigi.util import inherits

from tasks import analysis, predict

# Pipeline definition consisting of three tasks:
# Extraction -> Prediction -> Analysis
# Runs for a single data slice given by the time interval date_from - date_to.


class BaseGeminiTask(luigi.Task):  # pylint: disable=too-few-public-methods
    """Base task class."""

    date_from = luigi.DateParameter()
    date_to = luigi.DateParameter()
    artifact_folder = luigi.Parameter()
    config_file = luigi.Parameter()

    def get_artifact_folder(self):
        """Get folder where artifacts from running task are stored.

        Returns
        -------
        str
            Folder to store artifacts.
        """
        folder = os.path.join(
            self.artifact_folder,
            self.date_to.strftime("%Y-%m-%d"),  # pylint: disable=no-member
        )
        try:
            os.mkdir(folder)
        except OSError:
            return folder

        return folder


@inherits(BaseGeminiTask)
class DataExtraction(BaseGeminiTask):
    """Data Extraction task."""

    def run(self):
        """Run extraction task."""
        # read and parse default configuration
        _ = config_parser.read_config(self.config_file)

    def output(self):
        """Save extracted data output CSV file."""
        return luigi.LocalTarget(os.path.join(self.get_artifact_folder(), "data.csv"))


@inherits(BaseGeminiTask)
class Prediction(BaseGeminiTask):
    """Prediction task."""

    def requires(self):
        """Add extraction as dependency task."""
        return DataExtraction(self.date_from, self.date_to)

    def run(self):
        """Run prediction task."""
        # specify arguments
        args = config_parser.read_config(self.config_file)
        args.result_output = self.output().fn
        args.input = self.input().fn
        predict.main(args)

    def output(self):
        """Save prediction output results."""
        return luigi.LocalTarget(
            os.path.join(self.get_artifact_folder(), "results.csv")
        )


class Analysis(BaseGeminiTask):
    """Analysis task."""

    def requires(self):
        """Add prediction as dependency task."""
        return Prediction(self.date_from, self.date_to)

    def run(self):
        """Run analysis task."""
        # run both reports and log main metrics
        # prepare config
        config = config_parser.read_config(self.config_file)
        config.test = self.input().fn
        config.slice = ""
        config.report_full_path = self.output()[0].fn
        analysis.main(config)

        config.type = "performance"
        config.report_full_path = self.output()[1].fn
        analysis.main(config)

    def output(self):
        """Save output reports."""
        ds_file = os.path.join(self.get_artifact_folder(), "dataset_report.json")
        ml_file = os.path.join(self.get_artifact_folder(), "model_report.json")
        return luigi.LocalTarget(ds_file), luigi.LocalTarget(ml_file)
