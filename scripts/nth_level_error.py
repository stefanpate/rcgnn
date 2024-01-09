import pandas as pd
from collections import defaultdict
import os
from src.utils import load_embed, save_json, ensure_dirs
import numpy as np


'''
Set these
'''
db = 'swissprot'
embed_type = 'clean'

# save_to = ''
db_dir = f"../data/{db}/"
embed_dir = f"{db_dir}{embed_type}/"
embed_csv = f"{db_dir}{db}.csv"
n_levels = 4 # Levels of hierarchy in EC
ds = 1000
batch_size = 100 # For getting predicted ec labels
seed = 825
rng = np.random.default_rng(seed)

# Load id -> ec look-up table
id2ec = pd.read_csv(embed_csv, delimiter='\t')
id2ec.set_index('Entry', inplace=True)

# Load embeddings
print("Loading embeddings")
ecs = []
embeds = []
embed_idxs = defaultdict(lambda : defaultdict(list)) # {ec level: {ec number up to level:[idx1, ...]}} (idxs in embed_arr)
for i, elt in enumerate(os.listdir(embed_dir)[::ds]):
    id, this_embed = load_embed(embed_dir + elt)
    this_ec = id2ec.loc[id, 'EC number']
    if ';' in this_ec: # Multiple ecs, take first
        this_ec = this_ec.split(';')[0]

    ecs.append(np.array(this_ec.split('.')).astype('<U1')) # EC str -> arr
    embeds.append(this_embed)

    for j in range(n_levels):
        sub_key = '.'.join(this_ec.split('.')[:j+1])
        embed_idxs[j][sub_key].append(i)

embeds = np.vstack(embeds)
ecs = np.vstack(ecs)

# Get centroids of level 4 clusters
print('Getting level 4 centroids')
l4_ecs = [] 
l4_centroids = []
for this_l4_ec in embed_idxs[3].keys():
    this_embeds = embeds[embed_idxs[3][this_l4_ec]]
    ec_arr = np.array(this_l4_ec.split('.')).astype('<U1')

    # Catch incomplete ecs
    if len(ec_arr) < n_levels:
        rnd_append = np.array([str(rng.random()) for i in range(n_levels - len(ec_arr))])
        ec_arr = np.hstack((ec_arr, rnd_append))
    
    l4_ecs.append(ec_arr)
    l4_centroids.append(this_embeds.mean(axis=0))

l4_centroids = np.vstack(l4_centroids)
l4_ecs = np.array(l4_ecs)

# Get predicted ec label
print("Getting predicted labels")
pred_ecs = []
n_batches = embeds.shape[0] // batch_size
l4_centroids_expand = np.transpose(l4_centroids[np.newaxis, :, :], axes=(0,2,1)) # Transpose to (1, # features, # centroids)

# Batch process samples to save memory
for i in range(n_batches):
    if i == n_batches - 1:
        dist_to_centroids = np.sqrt(np.square(embeds[i * batch_size:, :, np.newaxis] - l4_centroids_expand).sum(axis=1))
    else:
        dist_to_centroids = np.sqrt(np.square(embeds[i * batch_size:(i + 1) * batch_size, :, np.newaxis] - l4_centroids_expand).sum(axis=1))
    this_pred_ecs = l4_ecs[np.argmin(dist_to_centroids, axis=1)]
    pred_ecs.append(this_pred_ecs)

    if i % 1000 == 0:
        print(f"Batch {i} / {n_batches}")

pred_ecs = np.vstack(pred_ecs)

# Get number accuracy for every ec at every level
# These have keys of all ecs at every level
# e.g., (1,), (1,1), (2,3,4,3)
n_correct = defaultdict(lambda : 0) # Number correct predictions
total = defaultdict(lambda : 0) # Total samples of that class
accuracy = {}

for l in range(n_levels):
    for i in range(pred_ecs.shape[0]):
        total[tuple(ecs[i, :l+1])] += 1
        if np.all(ecs[i, :l+1] == pred_ecs[i, :l+1]):
            n_correct[tuple(ecs[i, :l+1])] += 1

for k in total.keys():
    accuracy[k] = n_correct[k] / total[k]

# Compute chance for every ec at every level
# Recall the ecs in l4_ecs are all the unique
# level 4 ecs found in the training data (uniprot)
omega = len(l4_ecs)
chance = defaultdict(lambda : 0)
for l in range(n_levels):
    for i in range(l4_ecs.shape[0]):
        chance[tuple(l4_ecs[i, :l+1])] += 1

for k,v in chance.items():
    chance[k] = v / omega