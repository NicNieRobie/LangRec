import pandas as pd

dataset = 'STEAM'

parquet_path = f'data_store/{dataset.lower()}/interactions.parquet'

df = pd.read_parquet(parquet_path)
df.rename(columns={df.columns[0]: 'user_id:token', df.columns[1]: 'item_id:token', df.columns[2]: f'label:float'},
          inplace=True)
df.to_csv(f'dataset_inter/ctr/{dataset}/{dataset}.inter', sep='\t', index=False)