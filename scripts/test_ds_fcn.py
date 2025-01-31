import pandas as pd
import numpy as np

path = "/scratch/spn1560/hiec/sprhea_v3_folded_pt_ns/rcmcs/3fold/train_val_0.parquet"

def downsample_negatives(data: pd.DataFrame, neg_multiple: int, rng: np.random.Generator):
    neg_idxs = data[data['y'] == 0].index
    n_to_rm = len(neg_idxs) - (len(data[data['y'] == 1]) * neg_multiple)
    idx_to_rm = rng.choice(neg_idxs, n_to_rm, replace=False)
    data.drop(axis=0, index=idx_to_rm, inplace=True)

if __name__ == '__main__':
    rng = np.random.default_rng(seed=1234)
    df = pd.read_parquet(path)
    ds = downsample_negatives(df, 3, rng)
    print()