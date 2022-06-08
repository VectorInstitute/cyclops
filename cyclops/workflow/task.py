"""Pipeline module that defines different tasks that can be executed in a workflow."""

import argparse
import glob
import logging
import os

import luigi
import pandas as pd
from luigi.util import inherits

from codebase_ops import get_log_file_path
from cyclops import config
from cyclops.constants import MIMIC
from cyclops.processors.column_names import EVENT_NAME, EVENT_TIMESTAMP, EVENT_VALUE
from cyclops.processors.constants import EMPTY_STRING, UNDERSCORE
from cyclops.processors.events import normalise_events
from cyclops.processors.util import has_columns
from cyclops.query.interface import QueryInterface
from cyclops.utils.file import save_dataframe
from cyclops.utils.log import setup_logging
from cyclops.workflow.constants import NORMALISE, QUERY

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


class BaseTask(luigi.Task):
    """Base task class."""

    config_path = luigi.Parameter()
    output_folder = luigi.Parameter(default=None)

    def create_output_folder(self) -> None:
        """Create folder where output files from running task are stored."""
        os.makedirs(self.output_folder, exist_ok=True)

    def read_config(self) -> argparse.Namespace:
        """Read in config from config_path, override params with luigi input params.

        Returns
        -------
        argparse.Namespace
            Configuration stored in object.

        """
        cfg = config.read_config(self.config_path)
        if self.output_folder:
            cfg.output_folder = self.output_folder

        return cfg


@inherits(BaseTask)
class QueryTask(BaseTask):
    """Data querying task."""

    query_interface = luigi.Parameter()
    output_file = luigi.OptionalParameter("data.parquet")

    def run(self) -> None:
        """Run querying task."""
        LOGGER.info("Running query task!")

        if not isinstance(self.query_interface, QueryInterface):
            raise ValueError("Query task accepts a query interface.")

        path = os.path.join(self.output_folder, self.output_file)
        self.query_interface.save(path)  # pylint: disable=no-member
        self.output()
        self.query_interface.clear_data()  # pylint: disable=no-member

    def output(self):
        """Query data saved as parquet files."""
        return luigi.LocalTarget(self.output_file)


@inherits(BaseTask)
class NormalizeEventsTask(BaseTask):
    """Clean and normalise event data task."""

    def requires(self) -> luigi.Task:
        """Require that some queried data through cyclops.query API is available."""
        return QueryTask(config_path=self.config_path, output_folder=self.output_folder)

    def run(self) -> None:
        """Run normalise event data task."""
        LOGGER.info("Running normalise events task!")
        cfg = self.read_config()
        query_output_files = glob.glob(
            os.path.join(cfg.output_folder, QUERY + UNDERSCORE + "*.gzip")
        )
        for query_output_file in query_output_files:
            dataframe = pd.read_parquet(query_output_file)
            if has_columns(dataframe, [EVENT_NAME, EVENT_VALUE, EVENT_TIMESTAMP]):
                dataframe = normalise_events(dataframe, filter_recognised=True)
                file_name = os.path.splitext(
                    os.path.basename(query_output_file).replace(
                        QUERY + UNDERSCORE, EMPTY_STRING
                    )
                )[0]

                path = os.path.join(self.output_folder, file_name)
                save_dataframe(dataframe, path)

    def output(self):
        """Query data saved as parquet files."""
        cfg = self.read_config()
        output_files = glob.glob(
            os.path.join(cfg.output_folder, NORMALISE + UNDERSCORE + "*.gzip")
        )

        yield [luigi.LocalTarget(output_file) for output_file in output_files]


if __name__ == "__main__":
    luigi.build(
        [NormalizeEventsTask(config_path=MIMIC, output_folder="_test_workflow")],
        workers=1,
        local_scheduler=False,
    )
