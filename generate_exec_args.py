import os

import pandas as pd
import yaml

DATASETS = ['MOVIELENS', 'GOODREADS']
CODE_TYPES = ['id', 'sid']

VALID_METRICS = {
    'ctr': 'AUC',
    'seq': 'NDCG@10',
    'drec': 'NDCG@10'
}

METRICS = {
    'ctr': ['AUC'],
    'seq': ['NDCG@1', 'NDCG@5', 'NDCG@10', 'MRR', 'HR@5', 'HR@10'],
    'drec': ['NDCG@1', 'NDCG@5', 'NDCG@10', 'HR@5', 'HR@10']
}

def build_args_list(model_cfg, mode):
    results = []

    def build_base_args(model_name, dataset, model_type, metrics, task, kind, requires_cloud,
                        valid_metric=None, code_type=None):
        base = f'--model {model_name} --dataset {dataset} --mode {mode} --type {model_type} --metrics {metrics} --task {task}'
        if valid_metric:
            base += f' --valid_metric {valid_metric}'
        if code_type:
            base += f' --code_type {code_type} --rqvae_epochs 250'
        if mode == 'testtune':
            base += ' --use_lora'

            if task == 'ctr':
                base += ' --lora_r 32 --lora_alpha 128'
            else:
                base += ' --lora_r 128 --lora_alpha 128'

            base += ' --batch_size 64' if kind == 'small' else ' --batch_size 16'
        results.append({'args': base, 'requires_cloud': requires_cloud, 'mode': mode})

    for model_name, model_entries in model_cfg.items():
        entries = model_entries if isinstance(model_entries, list) else [model_entries]

        for entry in entries:
            task = entry.get('task')
            model_type = entry.get('type')
            kind = entry.get('kind')
            requires_cloud = kind == 'large' or mode != 'test'

            metrics_str = ','.join(METRICS[task])
            valid_metric = VALID_METRICS.get(task) if mode == 'testtune' else None

            if model_type == 'encoding':
                for dataset in DATASETS:
                    for code_type in CODE_TYPES:
                        is_cloud = requires_cloud or code_type == 'sid'
                        build_base_args(model_name, dataset, 'prompt', metrics_str, task, kind, is_cloud, valid_metric, code_type)
            else:
                for dataset in DATASETS:
                    build_base_args(model_name, dataset, model_type, metrics_str, task, kind, requires_cloud, valid_metric)

    return results


if __name__ == "__main__":
    with open('config/model_config.yaml', 'r') as f:
        model_config = yaml.safe_load(f)

    test_args = build_args_list(model_config, mode='test')
    tune_args = build_args_list(model_config, mode='testtune')

    all_args = test_args + tune_args

    df = pd.DataFrame(all_args)

    os.makedirs('datasphere_data', exist_ok=True)
    path = os.path.join('datasphere_data', 'tasks_args.csv')
    df.to_csv(path, index=False)
