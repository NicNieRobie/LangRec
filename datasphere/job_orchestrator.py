import json
import os
import re
import signal
import threading

from loguru import logger

from datasphere.job_runner import DataSphereJobRunner
from datasphere.utils.cmd import run_cmd
from datasphere.utils.telegram_notify import send_telegram_message, NotificationType

STATE_FILE = os.path.join('datasphere_data', 'jobs_state.json')
MAX_CONCURRENT_JOBS = 6
POLL_INTERVAL = 30


class DataSphereJobOrchestrator:
    def __init__(self):
        self.state = self._load_state()
        self.running_procs = {}

        self.runner = DataSphereJobRunner()

        self.exit_event = threading.Event()

    def stage_jobs(self, pending_jobs):
        self.state['pending'].clear()

        success_args_list = self._get_successful_or_running_jobs_args()

        jobs_to_be_added = [entry for entry in pending_jobs if entry['args'] not in success_args_list]

        logger.info(
            f"{len(pending_jobs) - len(jobs_to_be_added)}/{len(pending_jobs)} jobs have already been successfully finished and won't be run again")

        self.state['pending'].extend(jobs_to_be_added)

    @staticmethod
    def _load_state():
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                return json.load(f)

        return {
            "pending": [],
            "running": {},
            "finished": {},
            "jobs_data": {}
        }

    def _save_state(self):
        with open(STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=2)

    @staticmethod
    def _get_job_status(job_id: str):
        try:
            job_list_cmd = ['datasphere', 'project', 'job', 'get', '--id', job_id]

            process = run_cmd(job_list_cmd)

            lines = process.stdout.readlines()
            process.stdout.close()
            process.wait()

            header_line_index = None
            for i, line in enumerate(lines):
                if 'Status' in line and 'ID' in line:
                    header_line_index = i
                    break

            if header_line_index is None:
                raise ValueError("Could not find header line with 'Status' column")

            data_line_index = header_line_index + 2
            if data_line_index >= len(lines):
                raise ValueError("No data line found after header")

            data_line = lines[data_line_index]

            header = lines[header_line_index]

            status_start = header.index('Status')

            columns = list(re.finditer(r'\S+', header))
            positions = [m.start() for m in columns]

            status_col_idx = positions.index(status_start)

            if status_col_idx + 1 < len(positions):
                status_end = positions[status_col_idx + 1]
            else:
                status_end = len(header)

            status = data_line[status_start:status_end].strip()

            return status
        except Exception as e:
            logger.error(f"Error checking job status: {e}")
            return "ERROR"

    def _terminate_all(self):
        logger.info("Terminating all running jobs...")

        for job_id, process in self.running_procs.items():
            logger.info(f"Terminating job {job_id}")
            try:
                if os.name == 'nt':
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except Exception as e:
                logger.warning(f"Failed to terminate job {job_id}: {e}")

        logger.info("Waiting for jobs to exit...")

        for process in self.running_procs.values():
            process.wait()

        logger.info("All jobs terminated.")

    def _get_successful_or_running_jobs_args(self):
        jobs_data = self.state['jobs_data']
        success_job_ids = set([k for k, v in self.state['finished'].items() if v['success']])

        running_args = set([v for k, v in self.state['running'].items()])

        success_args_list = set(
            [jobs_data.get(job_id).get('args') for job_id in success_job_ids if job_id in jobs_data])

        return running_args.union(success_args_list)

    def _launch_job(self, args: str):
        success_or_running_args_list = self._get_successful_or_running_jobs_args()

        if args in success_or_running_args_list:
            logger.info(f'Job with args {args} already finished or running, skipping')
            return

        result = self.runner.run_job(args)

        if result['success']:
            job_id = result['job_id']
            task_id = result['task_id']
            process = result['proc']

            self.running_procs[job_id] = process
            self.state["running"][job_id] = args
            self.state["jobs_data"][job_id] = {
                "task_id": task_id,
                "args": args
            }
            self._save_state()
            send_telegram_message(NotificationType.LAUNCH, result)
        else:
            print(f"Failed to launch job for args: {args}")
            send_telegram_message(NotificationType.ERROR_LAUNCH, result)

    def run(self):
        try:
            while not self.exit_event.is_set():
                while len(self.running_procs) < MAX_CONCURRENT_JOBS and len(
                        self.state['running']) < MAX_CONCURRENT_JOBS and self.state["pending"]:
                    params_dict = self.state["pending"].pop(0)
                    self._launch_job(params_dict['args'])

                finished_jobs = []
                for job_id in list(self.state["running"].keys()):
                    status = self._get_job_status(job_id)

                    logger.info(f"Job {job_id} status: {status}")

                    if status == "SUCCESS":
                        logger.info(f"Job {job_id} finished successfully.")
                        finished_jobs.append({
                            'id': job_id,
                            'success': True
                        })
                        task_id = self.state["jobs_data"].get(job_id).get("task_id")
                        send_telegram_message(
                            NotificationType.SUCCESS,
                            {'job_id': job_id, 'task_id': task_id}
                        )
                    elif status == "ERROR":
                        logger.error(f"Job {job_id} exited with error. Stopping all jobs.")
                        finished_jobs.append({
                            'id': job_id,
                            'success': False
                        })
                        task_id = self.state["jobs_data"].get(job_id).get("task_id")
                        send_telegram_message(
                            NotificationType.ERROR_RUN,
                            {'job_id': job_id, 'task_id': task_id}
                        )

                        self.exit_event.set()

                        break
                    elif status == "CANCELLED":
                        logger.info(f"Job {job_id} cancelled. It won't be rerun automatically.")
                        finished_jobs.append({
                            'id': job_id,
                            'success': True
                        })

                for entry in finished_jobs:
                    job_id = entry['id']
                    success = entry['success']

                    self.state["finished"][job_id] = {
                        'success': success
                    }
                    self.state["running"].pop(job_id, None)

                    process = self.running_procs.pop(job_id, None)
                    if process and process.poll() is None:
                        process.terminate()

                    self._save_state()

                self.exit_event.wait(POLL_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Ctrl+C received. Cleaning up...")
            self.exit_event.set()

        finally:
            self.terminate()

    def terminate(self):
        self._terminate_all()
        self._save_state()
        logger.info("Shutdown complete.")
