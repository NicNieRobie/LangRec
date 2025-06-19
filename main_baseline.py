import os

from utils.argparser import ArgParser
from utils.seed import seed
from runner_baseline import BaselineRunner
from loguru import logger

if __name__ == '__main__':
    cli_config_path = os.environ.get('CLI_CONFIG_PATH', 'config/cli/baseline_cli_config.yaml')
    arg_parser = ArgParser(cli_config_path)
    config = arg_parser.parse_args()

    logger.debug(f'Args: {vars(config)}')

    seed(config.seed)

    runner = BaselineRunner(config)
    runner.run()
