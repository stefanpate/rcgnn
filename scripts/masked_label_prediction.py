import pandas as pd
from collections import defaultdict
import os
from src.utils import load_embed
import numpy as np


'''
Set these
'''
db = 'erxprot'
train_db = 'erxprot'
embed_type = 'clean'

save_acc = f"../artifacts/embed_analysis/masked_label_prediction_acc_train_{train_db}_test_{db}_{embed_type}.txt"
embed_dir = f"../data/{db}/{embed_type}/"
embed_csv = f"../data/{db}/{db}.csv"
train_dir = f"../data/{train_db}/{embed_type}/"
train_csv = f"../data/{train_db}/{train_db}.csv"
n_levels = 4 # Levels of hierarchy in EC
batch_size = 10 # For getting predicted ec labels
ds = 1

# Different key to pull tensor from .pt file
if train_db == 'erxprot':
	train_embed_key = 32
elif train_db == 'swissprot':
	train_embed_key = 33
     
if db == 'erxprot':
     embed_key = 32
else:
     embed_key = 33

# Load swissprot id -> ec look-up table
train_id2ec = pd.read_csv(train_csv, delimiter='\t')
train_id2ec.set_index('Entry', inplace=True)

# Load swissprot embeddings
print("Loading training data")
train_embeds = []
embed_idxs = defaultdict(lambda : defaultdict(list)) # {ec level: {ec number up to level:[idx1, ...]}} (idxs in embed_arr)
for i, elt in enumerate(os.listdir(train_dir)[::ds]):
    id, this_embed = load_embed(train_dir + elt, train_embed_key)
    this_ec = train_id2ec.loc[id, 'EC number']
    
    if ';' in this_ec: # Multiple ecs, take first
        this_ec = this_ec.split(';')[0]

    train_embeds.append(this_embed)

    # Append idxs for all sub-ecs of this embed
    for j in range(n_levels):
        sub_key = '.'.join(this_ec.split('.')[:j+1])
        embed_idxs[j][sub_key].append(i)

train_embeds = np.vstack(train_embeds)

# Load test dataset id -> ec look-up table
id2ec = pd.read_csv(embed_csv, delimiter='\t')
id2ec.set_index('Entry', inplace=True)

# Load test dataset embeddings
ecs = []
embeds = []
for i, elt in enumerate(os.listdir(embed_dir)):
    id, this_embed = load_embed(embed_dir + elt, embed_key)
    this_ec = id2ec.loc[id, 'EC number']
    
    if ';' in this_ec: # Multiple ecs, take first
        this_ec = this_ec.split(';')[0]

    ecs.append(np.array(this_ec.split('.')).astype(str)) # EC str -> arr
    embeds.append(this_embed)

embeds = np.vstack(embeds)
ecs = np.vstack(ecs)

mask_accuracy = [] # Store accuracy masking at all 4 levels
n_samples = embeds.shape[0]
for l in range(n_levels):
    # Get lth-level centroids
    l_ecs = [] 
    l_centroids = []
    for this_l_ec in embed_idxs[l]:
        this_embeds = train_embeds[embed_idxs[l][this_l_ec]]
        ec_arr = np.array(this_l_ec.split('.')).astype(str) 
        l_ecs.append(ec_arr)
        l_centroids.append(this_embeds.mean(axis=0))

    l_centroids = np.vstack(l_centroids)
    l_ecs = np.array(l_ecs)

    # Get predicted ec label
    # Batch process samples to save memory
    pred_ecs = []
    n_batches = embeds.shape[0] // batch_size
    l_centroids_expand = np.transpose(l_centroids[np.newaxis, :, :], axes=(0,2,1)) # Transpose to (1, # features, # centroids)
    for i in range(n_batches):
        if i == n_batches - 1:
            dist_to_centroids = np.sqrt(np.square(embeds[i * batch_size:, :, np.newaxis] - l_centroids_expand).sum(axis=1))
        else:
            dist_to_centroids = np.sqrt(np.square(embeds[i * batch_size:(i + 1) * batch_size, :, np.newaxis] - l_centroids_expand).sum(axis=1))
        
        this_pred_ecs = l_ecs[np.argmin(dist_to_centroids, axis=1)]
        pred_ecs.append(this_pred_ecs)

    # Compare predicted to actual
    pred_ecs = np.vstack(pred_ecs)
    this_acc = (np.all(pred_ecs == ecs[:,:l+1], axis=1)).astype(int).sum(axis=0) / n_samples
    mask_accuracy.append(this_acc)
    print("Done w/ level: ", l+1)

# Save
print("Saving")
with open(save_acc, 'w') as f:
    for elt in mask_accuracy:
        f.write(str(elt) + '\n')

print("Done")
