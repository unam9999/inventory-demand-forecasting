import logging
import os

def setup_logging(log_path='project.log'):
    """
    Sets up logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def ensure_directories(paths):
    """
    Ensures that the list of directories exists.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
