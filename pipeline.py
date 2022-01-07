import luigi
from luigi.util import inherits
from tasks.datapipeline.process_data import pipeline
import config.config as config_parser
import tasks.predict as predict
import tasks.analysis as analysis
import os

# Pipeline definition consisting of three tasks: Extraction -> Prediction -> Analysis
# Runs for a single data slice given by the time interval date_from - date_to


class BaseGeminiTask(luigi.Task):
    date_from = luigi.DateParameter()
    date_to = luigi.DateParameter()
    artifact_folder = luigi.Parameter()
    config_file = luigi.Parameter()

    def get_artifact_folder(self):
        folder = os.path.join(self.artifact_folder, self.date_to.strftime("%Y-%m-%d"))
        try:
            os.mkdir(folder)
        except OSError:
            return folder

        return folder


@inherits(BaseGeminiTask)
class DataExtraction(BaseGeminiTask):
    def run(self):
        # read and parse default configuration
        config = config_parser.read_config(self.config_file)
        config.filter_date_from = self.date_from
        config.filter_date_to = self.date_to
        config.output_full_path = self.output().fn
        config.w = True
        config.r = True
        pipeline(config)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_artifact_folder(), "data.csv"))


@inherits(BaseGeminiTask)
class Prediction(BaseGeminiTask):
    def requires(self):
        return DataExtraction(self.date_from, self.date_to)

    def run(self):
        # specify arguments
        args = config_parser.read_config(self.config_file)
        args.result_output = self.output().fn
        args.input = self.input().fn
        predict.main(args)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(self.get_artifact_folder(), "results.csv")
        )


class Analysis(BaseGeminiTask):
    def requires(self):
        return Prediction(self.date_from, self.date_to)

    def run(self):
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
        ds_file = os.path.join(self.get_artifact_folder(), "dataset_report.json")
        ml_file = os.path.join(self.get_artifact_folder(), "model_report.json")
        return luigi.LocalTarget(ds_file), luigi.LocalTarget(ml_file)
