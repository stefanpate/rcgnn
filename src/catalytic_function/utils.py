import pandas as pd
import scipy as sp
import json
import numpy as np
import torch
import os
import subprocess
from sklearn.model_selection import KFold
from collections import namedtuple
import re
import yaml

data_dir = "/projects/b1039/trw7841/hiec/data"
scratch_dir = "/scratch/trw7841"

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

def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def retrive_esm1b_embedding(fasta_path, outdir):
    esm_script = "/home/spn1560/hiec/src/esm/scripts/extract.py"
    esm_type = "esm1b_t33_650M_UR50S"
    command = ["python", esm_script, esm_type, 
              fasta_path, outdir, "--include", "mean"]
    subprocess.run(command)