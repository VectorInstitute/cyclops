"""Python logging function."""

import logging
from typing import Optional, Union

LOG_FORMAT = "%(asctime)-15s %(levelname)-5s %(name)-15s - %(message)s"

# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
COLORS = {
    "WARNING": YELLOW,
    "INFO": WHITE,
    "DEBUG": BLUE,
    "CRITICAL": YELLOW,
    "ERROR": RED,
}

# The background is set with 40 plus the number of the color, and the foreground 30.
# These are the sequences need to get colored ouput.
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"


def formatter_message(message: str, use_color: Optional[bool] = True) -> str:
    """Format message.

    Parameters
    ----------
    message: str
        Message format.
    use_color: bool, optional

    Returns
    -------
    str
        Formatted message.

    """
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message


class Formatter(logging.Formatter):
    """Formatter.

    Attributes
    ----------
    use_color: bool
        Flag to toggle use of color.

    """

    def __init__(
        self, msg: Optional[str] = formatter_message(LOG_FORMAT, True), use_color=True
    ):
        """Instantiate.

        Parameters
        ----------
        msg: str
            Message format.
        use_color: bool
            Flag to set using colored formatting.

        """
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Apply formatting.

        Parameters
        ----------
        record: logging.LogRecord
            Record object for logging.

        Returns
        -------
        str
            Formatted string for log.

        """
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = (
                COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            )
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)


def setup_logging(
    log_path: Optional[str] = None,
    log_level: Union[int, str] = "DEBUG",
    print_level: Union[int, str] = "INFO",
    logger: Optional[logging.Logger] = None,
    use_color: Optional[bool] = True,
):
    """Create logger, and set it up.

    Parameters
    ----------
    log_path : str, optional
        Path to output log file.
    log_level : str, optional
        Log level for logging, defaults to DEBUG.
    print_level : str, optional
        Print level for logging, defaults to INFO.
    logger : logging.Logger, optional
        Pass logger if already exists, else a new logger object is created.
    use_color: bool, optional
        Use colored logging.

    """
    fmt = formatter_message(LOG_FORMAT, use_color)
    logger = logger if logger else logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers = []

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(Formatter(fmt, use_color=use_color))
    stream_handler.setLevel(print_level)
    logger.addHandler(stream_handler)

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(Formatter(fmt, use_color=use_color))
        logger.addHandler(file_handler)
