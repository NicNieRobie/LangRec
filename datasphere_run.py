import os
import subprocess
import uuid as uuid

from utils.argparser import ArgParser
from utils.auth import DATASPHERE_PROJ

if __name__ == "__main__":
    cli_config_path = os.environ.get('CLI_CONFIG_PATH', 'config/cli/cli_config.yaml')
    arg_parser = ArgParser(cli_config_path)
    arg_parser.parse_args()

    datasphere_config_dir = os.environ.get('DATASPHERE_CONFIG_DIR', os.path.join('config', 'datasphere'))
    base_config_path = os.path.join(datasphere_config_dir, 'datasphere_config_template.yaml')

    with open(base_config_path, 'r') as f:
        datasphere_config_template = f.read()

    task_uuid = uuid.uuid4()

    config = datasphere_config_template.format(args=arg_parser.args_str, uuid=str(task_uuid))

    task_config_dir = os.path.join(datasphere_config_dir, 'tasks')

    os.makedirs(task_config_dir, exist_ok=True)

    task_config_name = f'datasphere_{task_uuid}.yaml'
    task_config_path = os.path.join(task_config_dir, task_config_name)

    with open(task_config_path, 'w', encoding='utf-8') as f:
        f.write(config)
        print(f'Task generated with UUID {str(task_uuid)}')

    if not DATASPHERE_PROJ:
        raise ValueError("Datasphere project key not found in auth config")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    command = ['datasphere', 'project', 'job', 'execute', '-p', DATASPHERE_PROJ, '-c', task_config_path]

    process = subprocess.Popen(
        command,
        cwd=current_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in process.stdout:
        print(line, end='')
