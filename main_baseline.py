import os

from utils.argparser import ArgParser
from utils.seed import seed
from runner_baseline import BaselineRunner

if __name__ == '__main__':
    cli_config_path = os.environ.get('CLI_CONFIG_PATH', 'config/cli/baseline_cli_config.yaml')
    arg_parser = ArgParser(cli_config_path)
    config = arg_parser.parse_args()

    print('Args:', vars(config))

    seed(config.seed)
    print("config:", config)

    runner = BaselineRunner(config)
    runner.run()