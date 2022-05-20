"""Pipeline module that defines different tasks that can be executed in a workflow."""

import argparse
import glob
import logging
import os

import luigi
from luigi.util import inherits

from codebase_ops import get_log_file_path
from cyclops import config
from cyclops.constants import MIMIC
from cyclops.utils.log import setup_logging
from cyclops.workflow.queries import QUERY_CATELOG

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


class BaseTask(luigi.Task):
    """Base task class."""

    artifact_folder = luigi.Parameter()
    config_path = luigi.Parameter()

    def create_artifact_folder(self) -> None:
        """Create folder where artifacts from running task are stored."""
        os.makedirs(self.artifact_folder, exist_ok=True)

    def read_config(self) -> argparse.Namespace:
        """Read in config from config_path.

        Returns
        -------
        argparse.Namespace
            Configuration stored in object.

        """
        return config.read_config(self.config_path)


@inherits(BaseTask)
class QueryTask(BaseTask):
    """Data querying task."""

    def run(self):
        """Run querying task."""
        LOGGER.info("Running query task!")
        self.read_config()
        query_gen_fn = QUERY_CATELOG["example_mimic_query"]
        for query_name, query_interface in query_gen_fn():
            query_interface.run()
            query_interface.save(folder_path=self.artifact_folder, file_name=query_name)
            query_interface.clear_data()

    def output(self):
        """Query data saved as parquet files."""
        output_files = glob.glob(os.path.join(self.artifact_folder, "*.gzip"))

        yield [luigi.LocalTarget(output_file) for output_file in output_files]


if __name__ == "__main__":
    luigi.build(
        [QueryTask(artifact_folder="test", config_path=MIMIC)],
        workers=1,
        local_scheduler=False,
    )
