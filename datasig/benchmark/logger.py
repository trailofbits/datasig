import logging
import sys
from logging import LogRecord


def colored_fmt(color: str, whole: bool = False) -> str:
    reset = "\x1b[0m"
    if whole:
        return f"{color}%(asctime)s - %(levelname)s - %(message)s{reset}"
    return f"%(asctime)s - {color}%(levelname)s{reset} - %(message)s"


# From https://stackoverflow.com/a/56944256
class ColoredFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[34m"
    green = "\x1b[32m"

    FORMATS = {
        logging.DEBUG: colored_fmt(blue),
        logging.INFO: colored_fmt(green),
        logging.WARNING: colored_fmt(yellow),
        logging.ERROR: colored_fmt(red, whole=True),
        logging.CRITICAL: colored_fmt(bold_red, whole=True),
    }

    def format(self, record: LogRecord):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColoredFormatter())
logger = logging.getLogger("datasig")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
