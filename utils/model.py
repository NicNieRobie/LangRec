import yaml

with open('config/model_config.yaml', 'r') as f:
    model_config = yaml.safe_load(f)


def match(key: str, task: str):
    key_normalized = key.strip().upper()
    task_normalized = task.strip()

    model_entry = model_config.get(key_normalized)
    if model_entry is None:
        return {}

    if isinstance(model_entry, list):
        for entry in model_entry:
            if str(entry.get("task", "")).strip() == task_normalized:
                return entry
    elif isinstance(model_entry, dict):
        if str(model_entry.get("task", "")).strip() == task_normalized:
            return model_entry

    return {}
