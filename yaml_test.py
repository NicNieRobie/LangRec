import os
import yaml

model_config_path = os.path.join('config', 'model_config.yaml')
with open(model_config_path, "r") as config_file:
    model_templates = yaml.safe_load(config_file)
    # print(model_templates)

    model_dict = {}
    for base_model in model_templates:
        if isinstance(model_templates[base_model], dict):
            for descriptor in model_templates[base_model]:
                model_dict[f"{base_model}{descriptor}"]=model_templates[base_model][descriptor]
        else:
            model_dict[base_model] = model_templates[base_model]

print(model_dict)
