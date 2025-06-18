import os
import platform
import subprocess
import sys


def run_cmd(cmd: list[str]):
    current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

    if platform.system() == "Windows":
        process = subprocess.Popen(
            cmd,
            cwd=current_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            text=True
        )
    else:
        process = subprocess.Popen(
            cmd,
            cwd=current_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            text=True
        )

    return process
