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
DATASPHERE_COM = auth.get('datasphere-community') or os.environ.get('DATASPHERE_COM')
TELEGRAM_BOT_TOKEN = auth.get('tg-bot-token') or os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = auth.get('tg-chat-id') or os.environ.get('TELEGRAM_CHAT_ID')

if not HF_KEY:
    raise RuntimeError("Hugging Face key not found.")
