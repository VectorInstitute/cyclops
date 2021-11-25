import config.config as config
import tasks.train as train
import tasks.analysis as analysis
import tasks.predict as predict
from tasks.datapipeline.process_data import pipeline as extract

# Top level to invoke specific operations 
if __name__ == "__main__":
    args = config.read_config()
    if args.extract:
        extract(args)
    else if args.train:
        train.main(args)
    else if args.predict:
        predict.main(args)
    else if args.analyze:
        analysis.main(args)
    else:
        raise ValueError("Operation is not specified")
        
