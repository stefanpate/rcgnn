import pandas as pd
import scipy as sp
import json
import numpy as np
import torch
import os
from sklearn.model_selection import KFold
from collections import namedtuple
import re
import yaml

data_dir = "/projects/p30041/spn1560/hiec/data"
scratch_dir = "/scratch/spn1560"

DatabaseEntry = namedtuple("DatabaseEntry", "db, id", defaults=[None, None])
Enzyme = namedtuple("Enzyme", "uniprot_id, sequence, ec, validation_score, existence, reviewed, organism", defaults=[None, None, None, None, None, None, None])

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

def construct_sparse_adj_mat(ds_name, toc):
        '''
        Returns sparse representation of sample x feature adjacency matrix
        and lookup of sample names from row idx key.

        Args
            - ds_name: Str name of dataset
            - toc: Table of contents csv
        Returns
            - adj: Sparse adjacency matrix
            - idx_sample: Index to label dict for samples
            - idx_feature: Index to label dict for features / classes
        '''
        try:
            df = pd.read_csv(f"../data/{ds_name}/{toc}.csv", delimiter='\t')
        except FileNotFoundError:
            df = pd.read_csv(f"./data/{ds_name}/{toc}.csv", delimiter='\t')

        # Load from dataset "table of contents csv"
        df.set_index('Entry', inplace=True)
        sample_idx = {}
        feature_idx = {}
        
        # Construct ground truth protein-function matrix
        print(f"Constructing {ds_name}:{toc} sparse adjacency matrix")
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

def get_sample_feature_idxs(ds_name, toc):
        '''
        Load in dicts mapping sample and feature labels
        to a standard indexing 

        Args:
            - toc: Table of contents csv
        '''      
        # Load from dataset "table of contents csv"
        df = pd.read_csv(f"../data/{ds_name}/{toc}.csv", delimiter='\t')
        df.set_index('Entry', inplace=True)
        sample_idx = {}
        feature_idx = {}
        
        # Construct ground truth protein-function matrix
        print(f"Loading {ds_name}:{toc} sample and feature idx dicts")
        for i, elt in enumerate(df.index):
            labels = df.loc[elt, 'Label'].split(';')
            sample_idx[elt] = i
            for label in labels:
                if label in feature_idx:
                    j = feature_idx[label]
                else:
                    j = len(feature_idx)
                    feature_idx[label] = j

        idx_sample = {v:k for k,v in sample_idx.items()}
        idx_feature = {v:k for k,v in feature_idx.items()}
            
        return idx_sample, idx_feature

def load_design_matrix(ds_name, toc, embed_type, sample_idx, do_norm=True, scratch_dir=scratch_dir, data_dir=data_dir):
        '''
        Args
            - ds_name: Str name of dataset
            - embed_type: Str
            - sample_idx: {sample_label : row_idx}
            - toc: Table of contents csv

        Returns
            - X: Design matrixs (samples x embedding dim)
        '''
        # Load from scratch if pre-saved
        path = f"{scratch_dir}/{ds_name}_{toc}_{embed_type}_X.npy"
        if os.path.exists(path):
            X = np.load(path)
        else:

            try:
                print(f"Loading {embed_type} embeddings for {ds_name}:{toc} dataset")
                magic_key = 33
                data_path = f"{data_dir}/{ds_name}/"
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
            except:
                print("Data not found in projects dir")

        return X



def split_data(
        X,
        y,
        ds_name,
        toc,
        n_splits,
        seed,
        neg_multiple=0,
        do_save=True
        ):
    '''
    Args
    ----
    X:list or array - All (sample, feature) index pairs of whole dataset
    y:list or array -  Labels of pairs in X
    ds_name:str - Dataset name
    toc:str - Name of table of contents i.e., csv file with (sample, feature, sequence)
    n_splits:int - Number cv splits
    seed:float - Random seed for this split
    neg_multiple:int - Multiple of positive samples there are of negative samples

    Returns
    -------
    split_guide:pandas df - entries of (train/test, split_idx, X1, X2, y)
        where
        train/test:str - 'train' or 'test' split
        split_idx:int - The cv data split index
        X1:int - smaple_idx e.g., protein_idx
        X2:int - label_idx e.g., reaction_idx
        y:int - (sample, label) pair hyperlabels, e.g., 0 or 1
    '''

    fn = f"{ds_name}_{toc}_{n_splits}_splits_{seed}_seed_{neg_multiple}_neg_multiple.csv"
    rng = np.random.default_rng(seed=seed)
    
    # Check if data split files already there
    if os.path.exists(f"{scratch_dir}/{fn}"):
        print(f"Found existing data splits for {ds_name} {toc} n_splits={n_splits} seed={seed}")
        split_guide = pd.read_csv(f"{scratch_dir}/{fn}", sep='\t')
        return split_guide

    if type(X) is list:
        X = np.array(X)
    if type(y) is list:
        y = np.array(y)

    # Infer adjacency mat dims
    n_rows = X[:,0].max()
    n_cols = X[:, 1].max()

    
    # Split provided data
    cols = ['train/test', 'split_idx', 'X1', 'X2', 'y']
    data = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (train_idx, test_idx) in enumerate(kfold.split(X)):

        # Sample negatives
        train_negs = _negative_sample_bipartite(int(len(train_idx) * neg_multiple), n_rows, n_cols, obs_pairs=X[train_idx], rng=rng)
        test_negs = _negative_sample_bipartite(int(len(test_idx) * neg_multiple), n_rows, n_cols, obs_pairs=X, rng=rng)

        # Stack negatives on positives
        X_train, y_train = np.vstack((X[train_idx], train_negs)), np.vstack((y[train_idx], np.zeros(shape=train_negs.shape[0]).reshape(-1,1)))
        X_test, y_test = np.vstack((X[test_idx], test_negs)), np.vstack((y[test_idx], np.zeros(shape=test_negs.shape[0]).reshape(-1,1)))

        # Shuffle within data splits
        p = rng.permutation(X_train.shape[0])
        X_train = X_train[p]
        y_train = y_train[p]

        # Append to data for split_guide
        train_rows = list(zip(['train' for _ in range(y_train.size)], [i for _ in range(y_train.size)], X_train[:, 0], X_train[:, 1], y_train.reshape(-1,)))
        test_rows = list(zip(['test' for _ in range(y_test.size)], [i for _ in range(y_test.size)], X_test[:, 0], X_test[:, 1], y_test.reshape(-1,)))
        data += train_rows
        data += test_rows

    split_guide = pd.DataFrame(data=data, columns=cols)

    if do_save:
        split_guide.to_csv(f"{scratch_dir}/{fn}", sep='\t', index=False)

    return split_guide

def load_data_split(
        ds_name,
        toc,
        sample_embed_type,
        n_splits,
        seed,
        idx_sample,
        idx_feature,
        split_idx,
        neg_multiple=0
        ):
    '''
    Args
    ----
    ds_name:str - Dataset name
    toc:str - Name of table of contents i.e., csv file with (sample, label, sequence)
    sample_embed_type:str
    n_splits:int - Number cv splits
    seed:float - Random seed for this split
    idx_sample:dict
    idx_feature:dict
    split_idx:int
    neg_multiple:int - Multiple of positive samples there are of negative samples

    Returns
    -------


    '''
    fn_split_guide_pref = f"{ds_name}_{toc}_{n_splits}_splits_{seed}_seed_{neg_multiple}_neg_multiple"
    fn_pref = fn_split_guide_pref + f"_{split_idx}_split_idx"
    fns = [fn_pref + suff for suff in ['_train.npy', '_test.npy']]
    
    if all([os.path.exists(f"{scratch_dir}/{fn}") for fn in fns]):
        train_data, test_data = [np.load(f"{scratch_dir}/{fn}") for fn in fns]

    else:

        split_guide = pd.read_csv(f"{scratch_dir}/{fn_split_guide_pref}.csv", sep='\t')
        train_split =  split_guide.loc[(split_guide['train/test'] == 'train') & (split_guide['split_idx'] == split_idx)]
        test_split =  split_guide.loc[(split_guide['train/test'] == 'test') & (split_guide['split_idx'] == split_idx)]

        sample_name = idx_sample[list(idx_sample.keys())[0]]
        test_embed = load_embed(f"{data_dir}/{ds_name}/{sample_embed_type}/{sample_name}.pt", embed_key=33)[1].numpy()
        d = test_embed.size
        tp = np.dtype([('sample_embed', np.float32, (d,)), ('feature', '<U100'), ('y', int)])
        tmp = []
        for split in [train_split, test_split]:
            samples = []
            features = []
            y = [elt for elt in split.loc[:, 'y']]
            for sample_idx in split.loc[:, 'X1']:
                sample_name = idx_sample[sample_idx]
                samples.append(load_embed(f"{data_dir}/{ds_name}/{sample_embed_type}/{sample_name}.pt", embed_key=33)[1])

            for feature_idx in split.loc[:, 'X2']:
                features.append(idx_feature[feature_idx])

            data = np.zeros(len(samples), dtype=tp)
            data['sample_embed'] = samples
            data['feature'] = features
            data['y'] = y
            tmp.append(data)

        train_data, test_data = tmp
        
        for (fn, data) in zip(fns, tmp):
            np.save(f"{scratch_dir}/{fn}", data)

        
    return train_data, test_data


def save_hps_to_scratch(hp, gs_name, hp_idx):
    with open(f"{scratch_dir}/{gs_name}_{hp_idx}_hp_idx.json", 'w') as f:
        json.dump(hp, f)

def load_hps_from_scratch(gs_name, hp_idx):
    with open(f"{scratch_dir}/{gs_name}_{hp_idx}_hp_idx.json", 'r') as f:
        hp = json.load(f)

    return hp

def load_known_rxns(path):
    with open(path, 'r') as f:
        data = json.load(f)

    for _,v in data.items():

        # Convert enzymes and db entries to namedtuples
        enzymes = []
        for e in v['enzymes']:
            for i in range(len(e)):
                if type(e[i]) == list: # Convert list ec to tuple for hashing, set ops
                    e[i]= tuple(e[i])

            enzymes.append(Enzyme(*e))

        v['enzymes'] = enzymes

        db_entries = [DatabaseEntry(*elt) for elt in v['db_entries']]
        v['db_entries'] = db_entries

    return data

def negative_sample_bipartite(n_samples, n_rows, n_cols, obs_pairs, seed):
    '''
    Samples n_samples negative pairs from an n_rows x n_cols adjacency matrix
    given obs_pairs, positive pairs which should not be sampled
    '''
    rng = np.random.default_rng(seed)

    if type(obs_pairs) == np.ndarray:
        obs_pairs = [tuple(obs_pairs[i, :]) for i in range(obs_pairs.shape[0])]
    
    # Sample subset of unobserved pairs
    unobs_pairs = []
    while len(unobs_pairs) < n_samples:
        i = rng.integers(0, n_rows)
        j = rng.integers(0, n_cols)

        if (i, j) not in obs_pairs:
            unobs_pairs.append((i, j))

    return np.array(unobs_pairs)

def _negative_sample_bipartite(n_samples, n_rows, n_cols, obs_pairs, rng):
    '''
    Samples n_samples negative pairs from an n_rows x n_cols adjacency matrix
    given obs_pairs, positive pairs which should not be sampled
    '''
    if type(obs_pairs) == np.ndarray:
        obs_pairs = [tuple(obs_pairs[i, :]) for i in range(obs_pairs.shape[0])]
    
    # Sample subset of unobserved pairs
    unobs_pairs = []
    while len(unobs_pairs) < n_samples:
        i = rng.integers(0, n_rows)
        j = rng.integers(0, n_cols)

        if (i, j) not in obs_pairs:
            unobs_pairs.append((i, j))

    return np.array(unobs_pairs)

def write_shell_script(
        allocation,
        partition,
        mem,
        time,
        file,
        arg_str,
        job_name
        ):
    
    '''
    Args
    ----
    allocation:str
        p30041 | b1039
    partition:str
        gengpu | short | normal | long | b1039
    mem:str
        memory required e.g., 8G
    time:str
        hours of compute e.g., 4
    file:str
        script name, e.g., file.py
    arg_str:
        string of args following script name e.g,. -d 4 --example
    '''
    
    # should_save = lambda x: " --save-gs-models" if x else '' # TODO backwards compat w/ MF
    
    shell_script = f"""#!/bin/bash
    #SBATCH -A {allocation}
    #SBATCH -p {partition}
    #SBATCH -N 1
    #SBATCH -n 1
    #SBATCH --mem={mem}
    #SBATCH -t {time}:00:00
    #SBATCH --job-name={job_name}
    #SBATCH --output=../logs/out/{job_name}
    #SBATCH --error=../logs/error/{job_name}
    #SBATCH --mail-type=END
    #SBATCH --mail-type=FAIL
    #SBATCH --mail-user=stefan.pate@northwestern.edu
    ulimit -c 0
    module load python/anaconda3.6
    module load gcc/9.2.0
    source activate hiec
    python -u {file} {arg_str}
    """
    shell_script = shell_script.replace("    ", "") # Remove tabs
    return shell_script

def read_last_ckpt(exp_dir):
    def get_step(chkpt_str):
        pattern = r'=(\d+)\.'
        match = re.search(pattern, chkpt_str)
        return int(match.group(1))

    versions = sorted([(fn, int(fn.split('_')[-1])) for fn in os.listdir(exp_dir)], key=lambda x : x[-1])
    latest_version = versions[-1][0]
    chkpts = sorted([(fn, get_step(fn)) for fn in os.listdir(f"{exp_dir}/{latest_version}/checkpoints")], key=lambda x : x[-1])
    latest_chkpt = chkpts[-1][0]

    return f"{exp_dir}/{latest_version}/{latest_chkpt}"