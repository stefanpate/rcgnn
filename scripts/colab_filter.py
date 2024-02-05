import pandas as pd
import numpy as np
from src.collaborative_filtering import CollaborativeFiltering
# from src.utils import ensure_dirs
import os
from sklearn.metrics import accuracy_score, roc_auc_score

dataset = 'swissprot'
y_fn = "protein_x_catalytic_function.npy"


rng = np.random.default_rng(seed=1234)
path = f"../data/{dataset}/"

# Read data table of contents csv
df = pd.read_csv(path + f"{dataset}.csv", delimiter='\t')
df.set_index('Entry', inplace=True)
entry_idxs = list(df.index)
ec_idxs = set()
for elt in df.loc[:, "EC number"]:
    for ec in elt.split(';'):
        ec_idxs.add(ec)
ec_idxs = list(ec_idxs)

# if os.path.exists(path + y_fn):
if False:
    print("Loading")
    y = np.load(path + y_fn)

else:
    # Construct ground truth protein-function matrix
    print("Constructing y")
    y = np.zeros(shape=(len(entry_idxs), len(ec_idxs)))
    x = 0
    for elt in df.index:
        ecs = df.loc[elt, 'EC number'].split(';')
        i = entry_idxs.index(elt)
        js = np.array([ec_idxs.index(ec) for ec in ecs])
        y[i, js] = 1
        x += 1
        print(f"{x}/{y.shape[0]}", end='\r')

    print("\nSaving y")
    np.save(path + y_fn, y)

n1, n0 = 0, 0
n_mask = (y.shape[0] * y.shape[1] * 0.01) // 2 # 50-50 split of 1s and 0s of 1% of elements

# Get mask rnd indices
print("Get ones to mask with method #1")
# Time-saving, space-spending
rnd_rows, rnd_cols = [], []
all_idxs = np.where(y == 1)
rnd_numbers = rng.integers(0, all_idxs[0].shape[0], size=(n_mask,))
rnd_rows += all_idxs[0][rnd_numbers]
rnd_cols += all_idxs[1][rnd_numbers]

mask_idxs = [rnd_rows, rnd_cols]


print("Get zeros to mask with method #2")
# Space-saving, time-spending
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

# Hold on to true values
mask_idxs = [np.array(elt) for elt in zip(*mask_idxs)]
y_true = y[mask_idxs[0], mask_idxs[1]]

y[mask_idxs[0], mask_idxs[1]] = 0 # Mask out

similarity = np.matmul(y, y.T) # Prot-prot similarity matrix

k = similarity.shape[1] # K-nearest-neighbors
threshes = np.sort(similarity, axis=1)[:, -k].reshape(-1,1)
similarity[similarity < threshes] = 0 # Zero out all but kNN
row_sum = similarity.sum(axis=1).reshape(-1,1)
similarity = np.divide(similarity, row_sum, out=np.zeros_like(similarity), where=row_sum!=0)

# Predict 
y_hat = np.matmul(similarity, y)
y_pred = y_hat[mask_idxs[0], mask_idxs[1]]

accuracy = accuracy_score(y_true, y_pred>0)
roc_auc = roc_auc_score(y_true, y_pred>0)

print(f"Accuracy: {accuracy}")
print(f"ROC AUC: {roc_auc}")


print("Done")

