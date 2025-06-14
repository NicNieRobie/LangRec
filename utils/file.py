import os


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
