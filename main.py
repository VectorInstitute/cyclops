"""Main script for running sub-tasks.

A main script to run sub-tasks like data extraction or model training/inference,
or a data drift experiment.

"""

from tasks import train
from tasks import analysis
from tasks import predict

from cyclops import config


# Top level to invoke specific operations
if __name__ == "__main__":
    args = config.read_config()
    if args.train:
        train.main(args)
    elif args.predict:
        predict.main(args)
    elif args.analyze:
        analysis.main(args)
    elif args.extract:
        pass
    else:
        raise ValueError("Operation is not specified")
