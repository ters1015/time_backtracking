import os
import logging
import datetime

def create_logger(output_dir, mode, name):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{mode}_{name}.log")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    return logger