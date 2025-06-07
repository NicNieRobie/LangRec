# TODO: Rewrite to yaml

with open('.model') as f:
    model_data = f.read()

model = {}
for line in model_data.strip().split('\n'):
    name, key = line.split('=')
    name = name.strip().lower()
    key = key.strip().split(',')
    key = [v.split(':') for v in key]
    key = {k.strip(): v.strip() for k, v in key}
    for k in key:
        model[f'{name}{k}'] = key[k]


def match(key):
    if key in model:
        return model[key]
    return None