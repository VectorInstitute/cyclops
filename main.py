import tasks.train as train
import tasks.analysis as analysis
import tasks.predict as predict
from tasks.datapipeline.process_data import pipeline as extract
import config.config as config


# Top level to invoke specific operations
if __name__ == "__main__":
    args = config.read_config()
    if args.extract:
        extract(args)
    elif args.train:
        train.main(args)
    elif args.predict:
        predict.main(args)
    elif args.analyze:
        analysis.main(args)
    else:
        raise ValueError("Operation is not specified")
