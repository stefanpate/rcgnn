import pandas as pd
import numpy as np
from src.collaborative_filtering import CollaborativeFiltering

rng = np.random.default_rng(seed=1234)
path = "../data/price/price.csv"
df = pd.read_csv(path, delimiter='\t') # Read data TOC csv
df.set_index('Entry', inplace=True)
entry_idxs = list(df.index)
ec_idxs = set()
for elt in df.loc[:, "EC number"]:
    for ec in elt.split(';'):
        ec_idxs.add(ec)
ec_idxs = list(ec_idxs)

# Construct ground truth protein-function matrix
y = np.zeros(shape=(len(entry_idxs), len(ec_idxs)))
x = 0
for elt in df.index:
    ecs = df.loc[elt, 'EC number'].split(';')
    i = entry_idxs.index(elt)
    js = np.array([ec_idxs.index(ec) for ec in ecs])
    y[i, js] = 1
    x += 1
    print(f"{x}/{y.shape[0]}", end='\r')
    # if x > 1000:
    #     break

n1, n0 = 0, 0
n_mask = (y.shape[0] * y.shape[1] * 0.01) // 2 # 50-50 split of 1s and 0s of 1% of elements
# rnd_rows, rnd_cols = [], []
# for i in range(2):
#     all_idxs = np.where(y == i)
#     rnd_numbers = rng.integers(0, all_idxs[0].shape[0], size=(n_mask,))
#     rnd_rows += all_idxs[0][rnd_numbers]
#     rnd_cols += all_idxs[1][rnd_numbers]

# mask_idxs = [rnd_rows, rnd_cols]



print('sdaf')
mask_idxs = []
while (n0 < n_mask) or (n1 < n_mask):
    print(f"n0:{n0}, n1:{n1}", end='\r')
    i, j = rng.integers(0, y.shape[0]), rng.integers(0, y.shape[1])

    if (y[i, j] == 1) and (n1 < n_mask):
        mask_idxs.append((i, j))
        n1 += 1
    elif (y[i, j] == 0) and (n0 < n_mask):
        mask_idxs.append((i, j))
        n0 += 1

y_true = [np.array(elt) for elt in zip(*mask_idxs)]


