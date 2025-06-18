import yaml

with open('config/model_config.yaml', 'r') as f:
    model_config = yaml.safe_load(f)


def match(key: str, task: str | None):
    key_normalized = key.strip().upper()

    if task is not None:
        task_normalized = task.strip()
    else:
        task_normalized = None

    model_entry = model_config.get(key_normalized)
    if model_entry is None:
        return {}

    if isinstance(model_entry, list):
        if task_normalized is None:
            return model_entry[0]

        for entry in model_entry:
            if str(entry.get("task", "")).strip() == task_normalized:
                return entry
    elif isinstance(model_entry, dict):
        if task_normalized is None:
            return model_entry

        if task_normalized is None or str(model_entry.get("task", "")).strip() == task_normalized:
            return model_entry

    return {}
