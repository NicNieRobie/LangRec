from loguru import logger
from tqdm import tqdm


def configure_logger():
    logger.remove()
    logger_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> [<level>{level: ^7}</level>] <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    logger.configure(handlers=[dict(sink=lambda msg: tqdm.write(msg, end=''), format=logger_format, colorize=True)])
