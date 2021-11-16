import luigi
from datapipeline.process_data import pipeline
import datapipeline.config as data_config
import predict
import analysis.analyis.main as analyze
import os

#constants - TODO change to parameters later
ARTIFACT_FOLDER = '../executions/'
MODEL_PATH = 'model.pt'
REFERENCE_DATA_PATH = '../reference.csv'
DATASET_CONFIG_FILE = 'config/gemini_data.cfg'
PREDICT_CONFIG_FILE = 'config/gemini_predict.cfg'
ANALYSIS_CONFIG_FILE = 'config/gemini_analysis.cfg'

class BaseGeminiTask(luigi.Task):
    date_from = luigi.DateParameter()
    date_to = luigi.DateParameter()
    artifact_folder = liugi.Parameter(default=ARTIFACT_FOLDER)

    def get_artifact_folder(self):
        folder = os.path.join(artifact_folder, self.date_to.strftime('%Y-%m-%d'))
        try:
            os.mkdir(folder)
        except:
            #folder must already exist
        return folder

class DataExtraction(BaseGeminiTask):
    data_config = liugi.Parameter(default=DATASET_CONFIG_FILE)
    output_location = liugi.Parameter(default=self.get_artifact_folder()+'data.csv')

    def run(self):
        #read and parse default configuration
        config = conf.read_config(data_config)
        config.filter_date_from  = self.date_from
        config.filter_date_to = self.date_to
        config.output = self.output_location
        pipeline(config)

    def output(self):
        return luigi.LocalTarget(self.output_location)

class Prediction(BaseGeminiTask):
    config = liugi.Parameter(default=PREDICT_CONFIG_FILE)
    output_location = liugi.Parameter(default=self.get_artifact_folder() + 'results.csv')

    def require(self):
        return DataExtraction(self.date_from, self.date_to)

    def run(self):
        # specify arguments
        args = predict.prepare_args(self.config)
        args.output = self.output_location
        args.input = self.input()
        predict.main(args)

    def output(self):
        return luigi.LocalTarget(self.output_location)

class Analysis(BaseGeminiTask):
    config = liugi.Parameter(default=ANALYSIS_CONFIG_FILE)

    def require(self):
        return Prediction(self.date_from, self.date_to)

    def run(self):
        # run both reports and log main metrics
        # prepare config
        config = analysis.read_config(data_config)
        config.reference = REFERENCE_DATA_PATH
        config.test = self.input()
        analysis.main(config)
        config.type = "performance"
        analysis.main(config)

    def output(self):
        return luigi.LocalTarget(self.get_artifact_folder() + 'ds_analysis.json'), luigi.LocalTarget(self.get_artifact_folder() + 'mdl_analysis.json')









