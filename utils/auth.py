import os

auth = {}

try:
    with open('.auth') as f:
        auth_data = f.read()
    for line in auth_data.strip().split('\n'):
        name, key = line.split('=')
        auth[name.strip()] = key.strip()
except FileNotFoundError:
    pass

HF_KEY = auth.get('hf') or os.environ.get('HF_KEY')
DATASPHERE_PROJ = auth.get('datasphere') or os.environ.get('DATASPHERE_PROJ')

if not HF_KEY:
    raise RuntimeError("Hugging Face key not found.")
