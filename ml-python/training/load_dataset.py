import pandas as pd
import pyarrow.parquet as pq
import glob

def load_features(path="data/features"):
    files = glob.glob(f"{path}/**/*.parquet", recursive=True)
    dfs = [pq.read_table(f).to_pandas() for f in files]
    return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    df = load_features()
    print(df.head())
