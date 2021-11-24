import config.config as config
import tasks.train as train
import tasks.analysis as analysis
import tasks.predict as predict
from tasks.datapipeline.process_data import pipeline as extract

# Top level to invoke operations, multiple operations can be specified
if __name__ == "__main__":
    args = config.read_config()
    if args.extract:
        extract(args)
    if args.train:
        train.main(args)
    if args.predict:
        predict.main(args)
    if args.analyze:
        analysis.main(args)

    if not args.extract and not args.train and not args.predict and not args.analyze:
        raise ValueError('No action specified')
        