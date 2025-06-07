import os

from utils.argparser import ArgParser
from runner import Runner

if __name__ == '__main__':
    cli_config_path = os.environ.get('CLI_CONFIG_PATH', 'config/cli_config.yaml')
    arg_parser = ArgParser(cli_config_path)
    config = arg_parser.parse_args()

    print('Args:', vars(config))

    runner = Runner(config)
