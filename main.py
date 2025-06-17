import os

from glob import glob

from loguru import logger

from runner import Runner
from utils.argparser import ArgParser
from utils.logger import configure_logger
from utils.seed import seed

if __name__ == '__main__':
    configure_logger()

    # This gets all the .yaml files that do not end with _config
    cli_config_path = set(glob(os.path.join('config', 'cli', '*.yaml'))) - set(glob(os.path.join('config', 'cli', '*_config.yaml')))
    arg_parser = ArgParser(cli_config_path)
    config = arg_parser.parse_args()

    logger.debug('Args:', vars(config))

    seed(config.seed)

    runner = Runner(config)
    runner.run()
