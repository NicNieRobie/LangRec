import os

from utils.argparser import ArgParser
from utils.seed import seed
from runner import Runner

if __name__ == '__main__':
    cli_config_path = os.environ.get('CLI_CONFIG_PATH', os.path.join('config', 'cli', 'cli_config.yaml'))
    arg_parser = ArgParser(cli_config_path)
    config = arg_parser.parse_args()

    print('Args:', vars(config))

    seed(config.seed)

    runner = Runner(config)
    runner.run()
