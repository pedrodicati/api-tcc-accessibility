import logging
import sys

def logger():
    # Set higher level for external libs logging
    logging.getLogger('botocore').setLevel(logging.ERROR)
    logging.getLogger('boto3').setLevel(logging.ERROR)
    logging.getLogger('urllib').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)

    LEVEL = "INFO"

    log_format = '%(asctime)s - %(levelname)s - %(filename)s[%(lineno)s] %(message)s'
    logger = logging.getLogger()
    for h in logger.handlers:
        logger.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)
    logger.setLevel(LEVEL)

    # save log to file
    handler = logging.FileHandler("log.log")
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)

    return logger