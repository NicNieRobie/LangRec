import pandas as pd

dataset = 'GOODREADS'

parquet_path = f'data_store/{dataset.lower()}/users.parquet'

df = pd.read_parquet(parquet_path)

records = []
for _, row in df.iterrows():
    uid = row.iloc[0]
    items = row.iloc[1]
    for i, item_id in enumerate(items):
        records.append([uid, item_id, i + 1])

inter_df = pd.DataFrame(records, columns=['user_id:token', 'item_id:token', 'timestamp:float'])

inter_df.to_csv(f'dataset_inter/seq/{dataset}/{dataset}.inter', sep='\t', index=False)
