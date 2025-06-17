import os

from loguru import logger

from runner import Runner
from utils.argparser import ArgParser
from utils.logger import configure_logger
from utils.seed import seed

if __name__ == '__main__':
    configure_logger()

    cli_config_path = os.environ.get('CLI_CONFIG_PATH', os.path.join('config', 'cli', 'cli_config.yaml'))
    arg_parser = ArgParser(cli_config_path)
    config = arg_parser.parse_args()

    logger.debug('Args:', vars(config))

    seed(config.seed)

    runner = Runner(config)
    runner.run()
