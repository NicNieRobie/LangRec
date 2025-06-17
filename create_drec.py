import pandas as pd

dataset = 'STEAM'

parquet_path = f'data_store/{dataset.lower()}/users.parquet'
user_history_df = pd.read_parquet(parquet_path)

parquet_path = f'data_store/{dataset.lower()}/interactions.parquet'
interactions = pd.read_parquet(parquet_path)

user_history_df = user_history_df.rename(columns={
    user_history_df.columns[0]: 'user_id',
    user_history_df.columns[1]: 'history',
})
interactions = interactions.rename(columns={
    interactions.columns[0]: 'user_id',
    interactions.columns[1]: 'item_id',
    interactions.columns[2]: 'label',
})

user_history_dict = dict(zip(user_history_df['user_id'], user_history_df['history']))

interactions = interactions.sort_values(by=['user_id'])

data = []

for user_id, group in interactions.groupby('user_id'):
    group = group.reset_index(drop=True)

    positives = group[group['label'] == 1]['item_id'].tolist()
    if not positives:
        continue

    label_item = positives[-1]

    negatives = group[group['label'] == 0]['item_id'].tolist()
    history_items = set(user_history_dict.get(user_id, []))

    filtered_negatives = [item for item in negatives if item not in history_items]

    data.append({
        'user_id:token': user_id,
        'candidates:token_seq': filtered_negatives,
        'label:float': label_item
    })

result_df = pd.DataFrame(data)

result_df.to_csv(f'dataset_inter/drec/{dataset}/{dataset}.inter', sep='\t', index=False)