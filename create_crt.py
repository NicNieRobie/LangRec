import pandas as pd
import os

dataset = 'GOODREADS'

repr = 'text'

parquet_path = f'data_store/{dataset.lower()}/interactions.parquet'

if repr == 'text':
    df = pd.read_parquet(parquet_path)
    df.rename(columns={df.columns[0]: 'user_id:token', df.columns[1]: 'item_id:token', df.columns[2]: f'label:float'},
              inplace=True)

    items = pd.read_parquet(f'data_store/{dataset.lower()}/items.parquet')
    items.rename(columns={items.columns[0]: 'item_id:token', items.columns[1]: 'label:float'},
              inplace=True)
    df = df.merge(items, on='item_id:token', how='inner')

    output_dir = f'dataset_inter/{repr}/ctr/{dataset}/'
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(f'{output_dir}{dataset}.inter', sep='\t', index=False)

elif repr == 'sem_id':
    pass

else:
    df = pd.read_parquet(parquet_path)
    df.rename(columns={df.columns[0]: 'user_id:token', df.columns[1]: 'item_id:token', df.columns[2]: f'label:float'},
              inplace=True)

    output_dir = f'dataset_inter/{repr}/ctr/{dataset}/'
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(f'{output_dir}{dataset}.inter', sep='\t', index=False)