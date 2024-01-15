import sys
sys.path.append('/home/spn1560/hiec/src')
import pandas as pd
from collections import defaultdict
import os
from src.utils import load_embed, save_json
import numpy as np


'''
Set these
'''
db = 'erxprot'
train_db = 'erxprot'
embed_type = 'clean'

save_acc = f"../artifacts/embed_analysis/nth_level_accuracy_train_{train_db}_test_{db}_{embed_type}.json"
save_tot = f"../artifacts/embed_analysis/nth_level_totals_train_{train_db}_test_{db}_{embed_type}.json"
save_chance = f"../artifacts/embed_analysis/nth_level_chance_train_{train_db}_test_{db}_{embed_type}.json"
embed_dir = f"../data/{db}/{embed_type}/"
embed_csv = f"../data/{db}/{db}.csv"
train_dir = f"../data/{train_db}/{embed_type}/"
train_csv = f"../data/{train_db}/{train_db}.csv"
n_levels = 4 # Levels of hierarchy in EC
batch_size = 10 # For getting predicted ec labels
ds = 1 # Downsample

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
print("Loading train data")
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

	if i % 100 == 0:
		print(f"{i}th training batch loaded")

train_embeds = np.vstack(train_embeds)

# Get centroids of level 4 clusters
print('Getting level 4 centroids')
l4_ecs = [] 
l4_centroids = []
for this_l4_ec in embed_idxs[n_levels - 1].keys():
	this_embeds = train_embeds[embed_idxs[3][this_l4_ec]]
	ec_arr = np.array(this_l4_ec.split('.')).astype(str)    
	l4_ecs.append(ec_arr)
	l4_centroids.append(this_embeds.mean(axis=0))

l4_centroids = np.vstack(l4_centroids)
l4_ecs = np.array(l4_ecs)

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

# Get predicted ec label
# Batch process samples to save memory
print("Getting predicted labels")
pred_ecs = []
n_batches = embeds.shape[0] // batch_size
l4_centroids_expand = np.transpose(l4_centroids[np.newaxis, :, :], axes=(0,2,1)) # Transpose to (1, # features, # centroids)
for i in range(n_batches):
	
	if i == n_batches - 1:
		dist_to_centroids = np.sqrt(np.square(embeds[i * batch_size:, :, np.newaxis] - l4_centroids_expand).sum(axis=1))
	else:
		dist_to_centroids = np.sqrt(np.square(embeds[i * batch_size:(i + 1) * batch_size, :, np.newaxis] - l4_centroids_expand).sum(axis=1))
	
	this_pred_ecs = l4_ecs[np.argmin(dist_to_centroids, axis=1)]
	pred_ecs.append(this_pred_ecs)

	if i % 20 == 0:
		print(f"Batch {i} / {n_batches}")

pred_ecs = np.vstack(pred_ecs)

print("Calculating accuracy")
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

# Convert keys to string
old_list = [accuracy, total, chance]
new_list = []
for elt in old_list:
	temp = {}
	for k,v in elt.items():
		this_k = '.'.join([str(dig) for dig in k])
		temp[this_k] = v

	new_list.append(temp)

accuracy, total, chance = new_list

# Save
print("Saving")
save_json(accuracy, save_acc)
save_json(total, save_tot)
save_json(chance, save_chance)
print("Done")
