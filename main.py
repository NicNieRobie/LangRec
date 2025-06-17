import os

from glob import glob

from utils.argparser import ArgParser
from utils.seed import seed
from runner import Runner

if __name__ == '__main__':
    # cli_config_path = os.environ.get('CLI_CONFIG_PATH', os.path.join('config', 'cli', 'cli_config.yaml'))

    # This gets all the .yaml files that do not end with _config
    cli_config_path = set(glob(os.path.join('config', 'cli', '*.yaml'))) - set(glob(os.path.join('config', 'cli', '*_config.yaml')))

    arg_parser = ArgParser(cli_config_path)
    config = arg_parser.parse_args()

    print('Args:', vars(config))

    seed(config.seed)

    runner = Runner(config)
    runner.run()
