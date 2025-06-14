import os

from data.encoding.sid_encoder import SIDEncoder
from utils.argparser import ArgParser
from utils.gpu import get_device

if __name__ == "__main__":
    cli_config_path = os.environ.get('SID_TEST_CLI_CONFIG_PATH', 'config/cli/sid_test_cli_config.yaml')
    config = ArgParser(cli_config_path).parse_args()
    device = get_device(config.gpu)

    sid_encoder = SIDEncoder(config, device)

    result = sid_encoder.encode()

    for i, (k, v) in enumerate(result.items()):
        print(k, v)
        if i == 4:
            break
