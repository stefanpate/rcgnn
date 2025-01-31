import pandas as pd
import scipy as sp
import json
import numpy as np
import torch
import os
import subprocess
from collections import namedtuple
from pathlib import Path
from omegaconf import OmegaConf

filepaths = OmegaConf.load("../configs/filepaths/base.yaml")
DatabaseEntry = namedtuple("DatabaseEntry", "db, id", defaults=[None, None])
Enzyme = namedtuple("Enzyme", "uniprot_id, sequence, ec, validation_score, existence, reviewed, organism", defaults=[None, None, None, None, None, None, None])

def save_json(data, save_to):
    with open(save_to, 'w') as f:
        json.dump(data, f)
    
def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def load_embed(path: Path, embed_key: int):
    id = path.stem
    f = torch.load(path)
    if type(f) == dict:
        embed = f['mean_representations'][embed_key]
    elif type(f) == torch.Tensor:
        embed = f
    return id, embed

def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def construct_sparse_adj_mat(path: Path):
        '''
        Returns sparse representation of sample x feature adjacency matrix
        and lookup of sample names from row idx key.

        Args
            - path: Path to dataset table of contents csv
        Returns
            - adj: Sparse adjacency matrix
            - idx_sample: Index to label dict for samples
            - idx_feature: Index to label dict for features / classes
        '''
        df = pd.read_csv(path, delimiter='\t')
        
        # Load from dataset "table of contents csv"
        df.set_index('Entry', inplace=True)
        sample_idx = {}
        feature_idx = {}
        
        # Construct ground truth protein-function matrix
        print(f"Constructing {path.stem} sparse adjacency matrix")
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

        adj = sp.sparse.csr_matrix((data, (row, col)), shape=(len(sample_idx), len(feature_idx)))
        idx_sample = {v:k for k,v in sample_idx.items()}
        idx_feature = {v:k for k,v in feature_idx.items()}
            
        return adj, idx_sample, idx_feature


# def load_precomputed_embeds(ds_name, toc, embed_type, sample_idx, do_norm=True, scratch_dir=scratch_dir, data_dir=data_dir):
#         '''
#         Args
#             - ds_name: Str name of dataset
#             - embed_type: Str
#             - sample_idx: {sample_label : row_idx}
#             - toc: Table of contents csv

#         Returns
#             - X: Design matrixs (samples x embedding dim)
#         '''
#         # Load from scratch if pre-saved
#         path = f"{scratch_dir}/{ds_name}_{toc}_{embed_type}_X.npy"
#         if os.path.exists(path):
#             X = np.load(path)
#         else:

#             try:
#                 print(f"Loading {embed_type} embeddings for {ds_name}:{toc} dataset")
#                 magic_key = 33
#                 data_path = f"{data_dir}/{ds_name}"
#                 X = [None for _ in range(len(sample_idx))]
#                 for i, (sample_id, idx) in enumerate(sample_idx.items()):
#                     X[idx] = load_embed(f"{data_path}/{embed_type}/{sample_id}.pt", embed_key=magic_key)[1]

#                     if i % 5000 == 0:
#                         print(f"Embedding #{i} / {len(sample_idx)}")

#                 X = np.vstack(X)
                
#                 if do_norm:
#                     X /= np.sqrt(np.square(X).sum(axis=1)).reshape(-1,1)

#                 # Save to scratch
#                 np.save(path, X)
#             except:
#                 raise ValueError("Data not found in projects dir")

#         return X

def load_embed_matrix(filepath: Path, idx_sample: dict, dataset: str, toc: str) -> np.ndarray:
    '''
    Given a filepath to embedding-containing dir on projects,
    checks if there is a numpy file containing an embedding
    matrix on scratch, if not creates it, finally loads it

    Args
    ----
    filepath:Path
        Filepath to directory containing individual embedding
        files as .pt
    idx_sample:dict
        Maps index of sample in embedding matrix / adjacency matrix
        to sample id
    dataset:str
        Name of dataset
    toc: str
        Name of table of contents
    

    Returns
    -------
    X:np.ndarray
        Embedding matrix (n_samples x d_embedding)
    '''
    rel_path = filepath.relative_to(filepaths["projects"])
    scratch_path = filepaths["scratch"] / rel_path / f"{dataset}_{toc}_X.npy"
    if scratch_path.exists():
        X = np.load(scratch_path)
    else:
        scratch_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Loading embeddings from {str(filepath)}")
        magic_key = 33
        X = [None for _ in range(len(idx_sample))]
        for i, (idx, sample_id) in enumerate(idx_sample.items()):
            X[idx] = load_embed(filepath / f"{sample_id}.pt", embed_key=magic_key)[1]

            if i % 5000 == 0:
                print(f"Embedding #{i} / {len(idx_sample)}")

        X = np.vstack(X)
        np.save(scratch_path, X)

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

def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def retrive_esm1b_embedding(fasta_path, outdir):
    esm_script = "/home/spn1560/hiec/src/esm/scripts/extract.py"
    esm_type = "esm1b_t33_650M_UR50S"
    command = ["python", esm_script, esm_type, 
              fasta_path, outdir, "--include", "mean"]
    subprocess.run(command)