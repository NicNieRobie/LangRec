import os

from utils.argparser import ArgParser
from utils.seed import seed
from runner_baseline import BaselineRunner
from loguru import logger

import numpy as np

if __name__ == '__main__':
    if not hasattr(np, 'float_'):
        np.float_ = np.float64
    if not hasattr(np, 'unicode_'):
        np.unicode_ = np.str_
    if not hasattr(np, 'unicode'):
        np.unicode = np.str_
    if not hasattr(np, 'complex_'):
        np.complex_ = np.complex128

    cli_config_path = os.environ.get('CLI_CONFIG_PATH', 'config/cli/baseline_cli_config.yaml')
    arg_parser = ArgParser(cli_config_path)
    config = arg_parser.parse_args()

    logger.debug(f'Args: {vars(config)}')

    seed(config.seed)

    runner = BaselineRunner(config)
    runner.run()
