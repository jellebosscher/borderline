import logging


def get_logger():
    # get logger for the project, if it does not exist, create it else return the existing one

    logger = logging.getLogger("aruco_detection")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
