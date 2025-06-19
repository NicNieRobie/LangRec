import pandas as pd
import os
from sentence_transformers import SentenceTransformer

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
    output_dir = f'dataset_inter/{repr}/drec/{dataset}/'
    items = pd.read_parquet(f'data_store/{dataset.lower()}/items.parquet')
    items.rename(columns={items.columns[0]: 'item_id:token', items.columns[1]: 'label:token'},
                 inplace=True)

    # Generate sentence embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    items['label_emb'] = model.encode(items['label:token'].astype(str).tolist()).tolist()

    # Create stable semantic IDs (e.g., simple hashed version)
    import hashlib


    def hash_id(x):
        return hashlib.md5(x.encode('utf-8')).hexdigest()[:12]


    items['sem_item_id:token'] = items['label:token'].apply(hash_id)

    # Save the semantic ID and embedding mapping
    items['label_emb'] = items['label_emb'].apply(lambda x: ' '.join(map(str, x)))
    item_output = items[['sem_item_id:token', 'label_emb']]
    item_output.columns = ['item_id:token', 'label_emb:float_seq']
    item_output.to_csv(f'{output_dir}{dataset}.item', sep='\t', index=False)

    # Map interactions to semantic IDs
    df = df.merge(items[['item_id:token', 'sem_item_id:token']], on='item_id:token', how='inner')
    df = df[['user_id:token', 'sem_item_id:token']]
    df.columns = ['user_id:token', 'item_id:token']  # final output format

    df.to_csv(f'{output_dir}{dataset}.inter', sep='\t', index=False)


output_dir = f'dataset_inter/{repr}/drec/{dataset}/'
os.makedirs(output_dir, exist_ok=True)

df.to_csv(f'{output_dir}{dataset}.inter', sep='\t', index=False)