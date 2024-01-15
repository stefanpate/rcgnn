import sys
sys.path.append('/home/spn1560/hiec/src')
import pandas as pd
from collections import defaultdict
import os
from src.utils import load_embed, save_json, ensure_dirs
import numpy as np
from src.evaluation import get_dmax, dmax_error


'''
Set these
'''
db = 'swissprot'
embed_type = 'clean'
seed = 825

'''
Main
'''
save_errors = f"../artifacts/embed_analysis/dmax_error_frac_{db}_{embed_type}.json"
save_dmaxes = f"../artifacts/embed_analysis/dmaxes_{db}_{embed_type}.json"
db_dir = f"../data/{db}/"
embed_dir = f"{db_dir}{embed_type}/"
embed_csv = f"{db_dir}{db}.csv"
n_levels = 4 # Levels of hierarchy in EC
do_shuffle = False
ds = 1 # Downsample factor
rng = np.random.default_rng(seed)

if db == 'erxprot':
     embed_key = 32
else:
     embed_key = 33

error_fracs = {}
dmaxes = {}
embed_idxs = defaultdict(lambda : defaultdict(list)) # {ec level: {ec number up to level:[idx1, ...]}} (idxs in embed_arr)

# Load id -> ec look-up table
id2ec = pd.read_csv(embed_csv, delimiter='\t')
id2ec.set_index('Entry', inplace=True)

# Load embeddings
print(f"Loading {db} embeddings")
ecs = []
embeds = []
for i, elt in enumerate(os.listdir(embed_dir)[::ds]):
    id, this_embed = load_embed(embed_dir + elt, embed_key)
    this_ec = id2ec.loc[id, 'EC number']
    if ';' in this_ec: # Multiple ecs, take first
        this_ec = this_ec.split(';')[0]

    ecs.append(np.array(this_ec.split('.')).astype(str)) # EC str -> arr
    embeds.append(this_embed)

    for j in range(n_levels):
        sub_key = '.'.join(this_ec.split('.')[:j+1])
        embed_idxs[j][sub_key].append(i)

embeds = np.vstack(embeds)
ecs = np.vstack(ecs)

# Shuffle data
if do_shuffle:
    print("Shuffling labels")
    rand_idxs = np.arange(embeds.shape[0])
    rng.shuffle(rand_idxs, axis=0) # Shuffle in place
    temp = embeds[rand_idxs]
    old_embeds = embeds
    embeds = temp

# Compute dmax errors
print("Computing dmaxes & dmax errors")
for i in range(n_levels):
    print(f"Level: {i+1}")
    error_fracs[i] = []
    dmaxes[i] = []
    for j in embed_idxs[i]:
        class_embeds = embeds[embed_idxs[i][j]]
        neg_embeds = np.delete(embeds, embed_idxs[i][j], axis=0)
        neg_ecs = np.delete(ecs, embed_idxs[i][j], axis=0)
        error_fracs[i].append(dmax_error(class_embeds, neg_embeds, neg_ecs))
        dmaxes[i].append(get_dmax(class_embeds))

# Save
print("Saving")
ensure_dirs('/'.join(save_errors.split('/')[:-1]))
save_json(error_fracs, save_errors)
save_json(dmaxes, save_dmaxes)

print("Done")
