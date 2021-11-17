import luigi
from luigi.util import inherits
from datapipeline.process_data import pipeline
import datapipeline.config as data_config
import predict
import analysis.analysis as analysis
import os

#constants - TODO change to parameters later
HOME = '/mnt/nfs/home/koshkinam/vector-delirium'
ARTIFACT_FOLDER = '/mnt/nfs/home/koshkinam/executions/'
MODEL_PATH = os.path.join(HOME,'model.pt')
REFERENCE_DATA_PATH = '/mnt/nfs/home/koshkinam/reference.csv'
DATASET_CONFIG_FILE = os.path.join(HOME, 'config/gemini_data.cfg')
PREDICT_CONFIG_FILE = os.path.join(HOME, 'config/gemini_predict.cfg')
ANALYSIS_CONFIG_FILE = os.path.join(HOME, 'config/gemini_analysis.cfg')

class BaseGeminiTask(luigi.Task):
    date_from = luigi.DateParameter()
    date_to = luigi.DateParameter()
    artifact_folder = luigi.Parameter(default=ARTIFACT_FOLDER)

    def get_artifact_folder(self):
        folder = os.path.join(self.artifact_folder, self.date_to.strftime('%Y-%m-%d'))
        try:
            os.mkdir(folder)
        except: 
            #folder must already exist
            return folder
        return folder

@inherits(BaseGeminiTask)
class DataExtraction(BaseGeminiTask):
    config_file = luigi.Parameter(default=DATASET_CONFIG_FILE)

    def run(self):
        #read and parse default configuration
        config = data_config.read_config(self.config_file)
        config.filter_date_from  = self.date_from
        config.filter_date_to = self.date_to
        config.output_full_path = self.output().fn
        config.w = True
        config.r = True
        pipeline(config)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_artifact_folder(), 'data.csv'))

@inherits(BaseGeminiTask)
class Prediction(BaseGeminiTask):
    config_file = luigi.Parameter(default=PREDICT_CONFIG_FILE)

    def requires(self):
        return DataExtraction(self.date_from, self.date_to)

    def run(self):
        # specify arguments
        args = predict.prepare_args(self.config_file)
        args.output = self.output().fn
        args.input = self.input().fn
        args.dataset_config = self.requires().config_file
        predict.main(args)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.get_artifact_folder(), 'results.csv'))

class Analysis(BaseGeminiTask):
    config_file = luigi.Parameter(default=ANALYSIS_CONFIG_FILE)

    def requires(self):
        return Prediction(self.date_from, self.date_to)

    def run(self):
        # run both reports and log main metrics
        # prepare config
        config = analysis.read_config(self.config_file)
        config.reference = REFERENCE_DATA_PATH
        config.test = self.input().fn
        config.slice = ''
        config.report_full_path = self.output()[0].fn
        analysis.main(config)

        config.type = "performance"
        config.report_full_path = self.output()[1].fn
        analysis.main(config)

    def output(self):
        ds_file = os.path.join(self.get_artifact_folder(), 'dataset_report.json')
        ml_file = os.path.join(self.get_artifact_folder(), 'model_report.json')
        return luigi.LocalTarget(ds_file), luigi.LocalTarget(ml_file)









