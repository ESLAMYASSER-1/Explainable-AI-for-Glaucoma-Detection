import logging
import logging.config
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,

    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        },
        "simple": {
            "format": "%(levelname)s | %(message)s"
        },
    },

    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": LOG_DIR / "app.log",
            "maxBytes": 5 * 1024 * 1024,  # 5 MB
            "backupCount": 3,
        },
    },

    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
}


def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
