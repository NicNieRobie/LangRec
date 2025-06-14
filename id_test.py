import os

from data.encoding.id_encoder import IDEncoder
from utils.argparser import ArgParser
from utils.gpu import get_device

if __name__ == "__main__":
    cli_config_path = os.environ.get('ID_TEST_CLI_CONFIG_PATH', 'config/cli/id_test_cli_config.yaml')
    config = ArgParser(cli_config_path).parse_args()
    device = get_device(config.gpu)

    id_encoder = IDEncoder(config, device)

    result = id_encoder.encode()

    for i, (k, v) in enumerate(result.items()):
        print(k, v)
        if i == 4:
            break
