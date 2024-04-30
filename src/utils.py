import pandas as pd
import scipy as sp
import json
import numpy as np
import torch
import os
from sklearn.model_selection import KFold

data_dir = "/projects/p30041/spn1560/hiec/data"
scratch_path = "/scratch/spn1560"

def save_json(data, save_to):
    with open(save_to, 'w') as f:
        json.dump(data, f)
    
def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def load_embed(path, embed_key):
    id = path.split('/')[-1].removesuffix('.pt')
    f = torch.load(path)
    if type(f) == dict:
        embed = f['mean_representations'][embed_key]
    elif type(f) == torch.Tensor:
        embed = f
    return id, embed

def load_class(ec, dir_path):
    '''Loads samples from
    provided EC class given 
    as an array or str'''
    if type(ec) == str:
        ec_str = '| ' + ec + '.'
    else:
        ec_str = '| ' + '.'.join([str(elt) for elt in ec]) + '.'
    
    ids, ecs, embeds = [], [], []
    for elt in os.listdir(dir_path):
        if ec_str in elt:
            path = dir_path + elt
            uni_id, this_ec, embed = load_embed(path)
            ids.append(uni_id)
            ecs.append(this_ec)
            embeds.append(embed)

    if len(embeds) > 0:
        embeds = torch.stack(embeds)

    return ids, ecs, embeds

def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def construct_sparse_adj_mat(ds_name):
        '''
        Returns sparse representation of sample x feature adjacency matrix
        and lookup of sample names from row idx key.

        Args
            - ds_name: Str name of dataset

        Returns
            -
        '''      
        # Load from dataset "table of contents csv"
        df = pd.read_csv(f"../data/{ds_name}/{ds_name}.csv", delimiter='\t')
        df.set_index('Entry', inplace=True)
        sample_idx = {}
        feature_idx = {}
        
        # Construct ground truth protein-function matrix
        print(f"Constructing {ds_name} sparse adjacency matrix")
        row, col, data = [], [], [] # For csr
        for i, elt in enumerate(df.index):
            labels = df.loc[elt, 'Label'].split(';')
            sample_idx[elt] = i
            for label in labels:
                if label in feature_idx:
                    j = feature_idx[label]
                else:
                    j = len(feature_idx)
                    feature_idx[label] = j
                row.append(i)
                col.append(j)
                data.append(1)
                
            print(f"{i}", end='\r')

        adj = sp.sparse.csr_matrix((data, (row, col)), shape=(len(sample_idx), len(feature_idx)))
        idx_sample = {v:k for k,v in sample_idx.items()}
        idx_feature = {v:k for k,v in feature_idx.items()}
            
        return adj, idx_sample, idx_feature

def load_design_matrix(ds_name, embed_type, sample_idx, do_norm=True):
        '''
        Args
            - ds_name: Str name of dataset
            - embed_type: Str
            - sample_idx: {sample_label : row_idx}

        Returns
            - X: Design matrixs (samples x embedding dim)
        '''
        # Load from scratch if pre-saved
        path = f"/scratch/spn1560/{ds_name}_{embed_type}_X.npy"
        if os.path.exists(path):
            X = np.load(path)
        else:

            print(f"Loading {embed_type} embeddings for {ds_name} dataset")
            magic_key = 33
            data_path = f"../data/{ds_name}/"
            X = []
            for i, elt in enumerate(sample_idx):
                X.append(load_embed(data_path + f"{embed_type}/{elt}.pt", embed_key=magic_key)[1])

                if i % 5000 == 0:
                    print(f"Embedding #{i} / {len(sample_idx)}")

            X = np.vstack(X)
            
            if do_norm:
                X /= np.sqrt(np.square(X).sum(axis=1)).reshape(-1,1)

            # Save to scratch
            np.save(path, X)

        return X

def split_data(ds_name, sample_embed_type, n_splits, seed):
    