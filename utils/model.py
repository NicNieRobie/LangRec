import yaml

with open('config/model_config.yaml', 'r') as f:
    model_config = yaml.safe_load(f)


def match(key: str):
    key = key.strip().lower()

    for model_name, variants_or_model in model_config.items():
        model_name_lower = model_name.lower()

        if isinstance(variants_or_model, dict):
            for variant, value in variants_or_model.items():
                variant_str = str(variant).lower()
                combined_key = f"{model_name_lower}{variant_str}"

                if combined_key == key:
                    if isinstance(value, dict):
                        keys = set(value.keys())

                        if len(keys) > 1:
                            return value

                        return value.get("path") or value.get("model_id") or value
                    return value

        else:
            if model_name_lower == key:
                return variants_or_model

    return None


if __name__ == "__main__":
    print(match('P5BEAUTY'))