import pandas as pd
import os

dataset = 'STEAM'

repr = 'text'

parquet_path = f'data_store/{dataset.lower()}/interactions.parquet'
df = pd.read_parquet(parquet_path)
df.rename(columns={df.columns[0]: 'user_id:token', df.columns[1]: 'item_id:token'},
          inplace=True)

if repr == 'text':
    items = pd.read_parquet(f'data_store/{dataset.lower()}/items.parquet')
    items.rename(columns={items.columns[0]: 'item_id:token', items.columns[1]: 'label:token'},
              inplace=True)
    df = df.merge(items, on='item_id:token', how='inner')
    df = df[['user_id:token', 'label:token']]

    df.rename(columns={df.columns[1]: 'item_id:token'},
              inplace=True)

elif repr == 'sem_id':
    pass

output_dir = f'dataset_inter/{repr}/drec/{dataset}/'
os.makedirs(output_dir, exist_ok=True)

df.to_csv(f'{output_dir}{dataset}.inter', sep='\t', index=False)