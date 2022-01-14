"""Script to run pipeline using Luigi."""

import luigi
import pipeline
from dateutil.relativedelta import relativedelta

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
        """[TODO: Add docstring]."""
        times = len(self.input())
        print(f"Ran analysis {times} times")
