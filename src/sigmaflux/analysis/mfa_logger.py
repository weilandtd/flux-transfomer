# custom_logger.py

import logging
import datetime

def setup_logging(log_to_file=False, log_to_console=True, level=logging.INFO, filename=None):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.propagate = False  # Avoid duplicate logging
    logging.getLogger('cobra').setLevel(logging.WARNING)


    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console Handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File Handler
    if log_to_file:
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f'log_{timestamp}.log'
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
