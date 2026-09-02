import logging
import sys

def setup_logging(level: int = logging.INFO):
    """
    Sets up structured logging for the application.
    """
    logging.basicConfig(
        level=level,
        format='%(levelname)s %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

