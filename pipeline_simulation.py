import luigi
import pipeline
from dateutil.relativedelta import relativedelta

class Simulation(luigi.Task): 
    date_from = luigi.DateParameter()
    date_to = luigi.DateParameter()

    def requires(self):
      results = []
      current_from = self.date_from
      current_to = current_from + relativedelta(months=1)
      while current_to <= self.date_to:
         results.append(pipeline.Analysis(current_from, current_to))
         current_to += relativedelta(months=1)
         current_from += relativedelta(months=1)
      return results

    def run(self):
       times = len(self.input())
       print(f"Ran analysis {times} times") 
