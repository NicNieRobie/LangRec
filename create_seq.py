import pandas as pd
import os

dataset = 'STEAM'

repr = 'text'

parquet_path = f'data_store/{dataset.lower()}/users.parquet'

df = pd.read_parquet(parquet_path)

records = []
for _, row in df.iterrows():
    uid = row.iloc[0]
    items = row.iloc[1]
    for i, item_id in enumerate(items):
        records.append([uid, item_id, i + 1])

df = pd.DataFrame(records, columns=['user_id:token', 'item_id:token', 'timestamp:float'])

if repr == 'text':

    items = pd.read_parquet(f'data_store/{dataset.lower()}/items.parquet')
    items.rename(columns={items.columns[0]: 'item_id:token', items.columns[1]: 'label:float'},
              inplace=True)
    df = df.merge(items, on='item_id:token', how='inner')

    output_dir = f'dataset_inter/{repr}/seq/{dataset}/'
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(f'{output_dir}{dataset}.inter', sep='\t', index=False)

elif repr == 'sem_id':
    pass

else:

    output_dir = f'dataset_inter/{repr}/seq/{dataset}/'
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(f'{output_dir}{dataset}.inter', sep='\t', index=False)