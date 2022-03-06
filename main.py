"""Main entrypoint script."""

from tasks import train
from tasks import analysis
from tasks import predict

import config


# Top level to invoke specific operations
if __name__ == "__main__":
    args = config.read_config()
    if args.train:
        train.main(args)
    elif args.predict:
        predict.main(args)
    elif args.analyze:
        analysis.main(args)
    else:
        raise ValueError("Operation is not specified")
