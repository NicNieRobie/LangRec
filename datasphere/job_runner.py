import os
import queue
import re
import threading
from glob import glob

from datasphere.utils.cmd import run_cmd
from utils.argparser import ArgParser
from utils.auth import DATASPHERE_PROJ
from utils.run import generate_run_name_and_hash

from loguru import logger


class DataSphereJobRunner:
    def __init__(self):
        cli_config_path = set(glob(os.path.join('config', 'cli', '*.yaml'))) - set(
            glob(os.path.join('config', 'cli', '*_config.yaml')))
        self.arg_parser = ArgParser(cli_config_path, validate_input=False)

        self.job_id_pattern = re.compile(r"created job `([a-z0-9]+)`")

    def run_job(self, args_str):
        args = self.arg_parser.parse_args(args_str)

        task_id, config_path = self._generate_job_config(args)

        return self._launch_job(task_id, config_path)

    def _generate_job_config(self, args):
        task_name, task_hash = generate_run_name_and_hash(args)

        datasphere_config_dir = os.environ.get('DATASPHERE_CONFIG_DIR', os.path.join('config', 'datasphere'))
        base_config_path = os.path.join(datasphere_config_dir, 'datasphere_config_template.yaml')

        with open(base_config_path, 'r') as f:
            datasphere_config_template = f.read()

        config = datasphere_config_template.format(args=self.arg_parser.args_str, name=str(task_name))

        task_config_dir = os.path.join(datasphere_config_dir, 'tasks')

        os.makedirs(task_config_dir, exist_ok=True)

        task_config_name = f'datasphere_{task_hash}.yaml'
        task_config_path = os.path.join(task_config_dir, task_config_name)

        with open(task_config_path, 'w', encoding='utf-8') as f:
            f.write(config)

        logger.info(f'Task generated with name {task_name} for args {args}')

        return task_name, task_config_path

    def _launch_job(self, task_id, task_config_path):
        if not DATASPHERE_PROJ:
            raise ValueError("Datasphere project key not found in auth config")

        run_job_command = ['datasphere', 'project', 'job', 'execute', '-p', DATASPHERE_PROJ, '-c', task_config_path]

        process = run_cmd(run_job_command)

        job_id_queue = queue.Queue()

        job_id_queue = queue.Queue()

        def stdout_reader():
            for line in process.stdout:
                match = self.job_id_pattern.search(line)
                if match:
                    job_id = match.group(1)
                    job_id_queue.put(job_id)
            process.stdout.close()

        t = threading.Thread(target=stdout_reader, daemon=True)
        t.start()

        try:
            job_id = job_id_queue.get(timeout=480)
            logger.info(f"Job for task {task_id} started with id: {job_id}")

            return {
                'success': True,
                'job_id': job_id,
                'task_id': task_id,
                'proc': process
            }
        except queue.Empty:
            logger.error(f"Failed to get job ID from output for task {task_id}")

            process.kill()
            return {
                'success': False,
                'task_id': task_id,
                'error': f'Failed to get job ID from output for task {task_id}'
            }
