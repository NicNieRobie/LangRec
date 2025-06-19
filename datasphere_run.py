import os
import subprocess
from glob import glob

from utils.argparser import ArgParser
from utils.auth import DATASPHERE_PROJ
from utils.run import generate_run_name_and_hash


def run_cmd(cmd: list[str]):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    process = subprocess.Popen(
        cmd,
        cwd=current_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    return process


def fetch_finished_jobs():
    job_list_cmd = ['datasphere', 'project', 'job', 'list', '-p', DATASPHERE_PROJ]

    process = run_cmd(job_list_cmd)

    stdout, stderr = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"Command failed: {stderr.strip()}")

    lines = stdout.strip().splitlines()[2:]
    job_names = [line.split()[1].strip() for line in lines if line.strip()][1:]

    return job_names


def generate_job_config(args):
    task_name, task_hash = generate_run_name_and_hash(args)

    print('Task name:', task_name)

    datasphere_config_dir = os.environ.get('DATASPHERE_CONFIG_DIR', os.path.join('config', 'datasphere'))
    base_config_path = os.path.join(datasphere_config_dir, 'datasphere_config_template.yaml')

    with open(base_config_path, 'r') as f:
        datasphere_config_template = f.read()

    config = datasphere_config_template.format(args=arg_parser.args_str, name=str(task_name))

    task_config_dir = os.path.join(datasphere_config_dir, 'tasks')

    os.makedirs(task_config_dir, exist_ok=True)

    task_config_name = f'datasphere_{task_hash}.yaml'
    task_config_path = os.path.join(task_config_dir, task_config_name)

    with open(task_config_path, 'w', encoding='utf-8') as f:
        f.write(config)
        print(f'Task generated with hash {str(task_hash)}')

    return task_name, task_config_path


def run_job(task_config_path):
    if not DATASPHERE_PROJ:
        raise ValueError("Datasphere project key not found in auth config")

    run_job_command = ['datasphere', 'project', 'job', 'execute', '-p', DATASPHERE_PROJ, '-c', task_config_path]

    process = run_cmd(run_job_command)

    for line in process.stdout:
        print(line, end='')


if __name__ == "__main__":
    cli_config_path = set(glob(os.path.join('config', 'cli', '*.yaml'))) - set(
        glob(os.path.join('config', 'cli', '*_config.yaml')))
    arg_parser = ArgParser(cli_config_path)
    config = arg_parser.parse_args()

    job_name, config_path = generate_job_config(config)

    finished_jobs = fetch_finished_jobs()

    if job_name in finished_jobs:
        response = input(
            f"[WARNING] Job '{job_name}' has already been completed. Do you want to continue? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            print("Aborting.")
            exit(0)

    run_job(config_path)
