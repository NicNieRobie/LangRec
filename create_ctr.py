import pandas as pd
import os

dataset = 'STEAM'

repr = 'sem_id'

parquet_path = f'data_store/{dataset.lower()}/interactions.parquet'

if repr == 'text':
    df = pd.read_parquet(parquet_path)
    df.rename(columns={df.columns[0]: 'user_id:token', df.columns[1]: 'item_id:token', df.columns[2]: f'label:float'},
              inplace=True)

    items = pd.read_parquet(f'data_store/{dataset.lower()}/items.parquet')
    items.rename(columns={items.columns[0]: 'item_id:token', items.columns[1]: 'item_label:token'},
              inplace=True)

    df = df.merge(items, on='item_id:token', how='inner')

    df = df[['user_id:token', 'item_label:token', 'label:float']]
    df = df.rename(columns={'item_label:token': 'item_id:token'})

    output_dir = f'dataset_inter/{repr}/ctr/{dataset}/'
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(f'{output_dir}{dataset}.inter', sep='\t', index=False)

elif repr == 'sem_id':
    output_dir = f'dataset_inter/{repr}/ctr/{dataset}/'
    os.makedirs(output_dir, exist_ok=True)

    # Load interactions
    df = pd.read_parquet(parquet_path)
    df.rename(columns={
        df.columns[0]: 'user_id:token',
        df.columns[1]: 'item_id:token',
        df.columns[2]: 'label:float'
    }, inplace=True)

    # Load and process item features
    items = pd.read_parquet(f'data_store/{dataset.lower()}/items.parquet')

    items.rename(columns={
        items.columns[0]: 'item_id:token',
        items.columns[1]: 'item_label:token'
    }, inplace=True)

    # Generate semantic embeddings for items
    from sentence_transformers import SentenceTransformer
    import hashlib

    model = SentenceTransformer('all-MiniLM-L6-v2')
    items['item_emb'] = model.encode(items['item_label:token'].astype(str).tolist()).tolist()

    # Create stable semantic IDs
    def hash_id(x):
        return hashlib.md5(x.encode('utf-8')).hexdigest()[:12]

    items['sem_item_id:token'] = items['item_id:token'].apply(hash_id)
    items['item_emb'] = items['item_emb'].apply(lambda x: ' '.join(map(str, x)))

    # Save item file
    item_output = items[['sem_item_id:token', 'item_emb']]
    item_output.columns = ['item_id:token', 'item_emb:float_seq']
    item_output.to_csv(f'{output_dir}{dataset}.item', sep='\t', index=False)

    # Map interactions to semantic IDs
    df = df.merge(items[['item_id:token', 'sem_item_id:token']], on='item_id:token', how='inner')
    df = df[['user_id:token', 'sem_item_id:token', 'label:float']]
    df.columns = ['user_id:token', 'item_id:token', 'label:float']
    df.to_csv(f'{output_dir}{dataset}.inter', sep='\t', index=False)

else:
    df = pd.read_parquet(parquet_path)
    df.rename(columns={df.columns[0]: 'user_id:token', df.columns[1]: 'item_id:token', df.columns[2]: f'label:float'},
              inplace=True)

    output_dir = f'dataset_inter/{repr}/ctr/{dataset}/'
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(f'{output_dir}{dataset}.inter', sep='\t', index=False)