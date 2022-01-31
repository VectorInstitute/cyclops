"""Script to run pipeline using Luigi."""

import logging
from dateutil.relativedelta import relativedelta
import luigi

import cyclops.workflow.pipeline as pipeline
from cyclops.utils.log import setup_logging, LOG_FILE_PATH


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=LOG_FILE_PATH, print_level="INFO", logger=LOGGER)


# Simulation of continuous pipeline running on regular intervals (monthly)
# for every interval between date_from to date_to


class Simulation(luigi.Task):
    """Simulation class."""

    date_from = luigi.DateParameter()
    date_to = luigi.DateParameter()

    def requires(self):
        """[TODO: Add docstring]."""
        results = []
        current_from = self.date_from
        current_to = current_from + relativedelta(months=1)
        while current_to <= self.date_to:
            results.append(pipeline.Analysis(current_from, current_to))
            current_to += relativedelta(months=1)
            current_from += relativedelta(months=1)
        return results

    def run(self):
        """Run simulation."""
        times = len(self.input())
        LOGGER.info(f"Ran analysis {times} times")
