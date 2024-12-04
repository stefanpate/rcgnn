from src.utils import construct_sparse_adj_mat, load_embed, load_json
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
from itertools import product, chain
import json
from collections import defaultdict
import subprocess
from dataclasses import dataclass, asdict, fields
from typing import List, Tuple
# from src.filepaths import filepaths
from pathlib import Path

def load_data_split(split_idx: int, scratch_path: Path = Path(''), dataset: str = '', data_path: Path = Path(''), idx_sample: dict = {}, idx_feature: dict = {}, split_guide: pd.DataFrame = None):
    # TODO: make better. Eliminate idx_feature & idx_sample by storing what you want in the .npy
    embed_dim = 1280
    embed_type = 'esm'

    if split_guide is None:
        train_data = np.load(scratch_path / f"{split_idx}_train.npy")
        test_data = np.load(scratch_path / f"{split_idx}_test.npy")
    else:
        train_split =  split_guide.loc[(split_guide['train/test'] == 'train') & (split_guide['split_idx'] == split_idx)]
        test_split =  split_guide.loc[(split_guide['train/test'] == 'test') & (split_guide['split_idx'] == split_idx)]

        tp = np.dtype([('sample_embed', np.float32, (embed_dim,)), ('feature', '<U100'), ('y', int)])
        tmp = []
        for split in [train_split, test_split]:
            samples = []
            features = []
            y = [elt for elt in split.loc[:, 'y']]
            for sample_idx in split.loc[:, 'X1']:
                sample_name = idx_sample[sample_idx]
                samples.append(load_embed(data_path / f"{dataset}/{embed_type}/{sample_name}.pt", embed_key=33)[1])

            for feature_idx in split.loc[:, 'X2']:
                features.append(idx_feature[feature_idx])

            data = np.zeros(len(samples), dtype=tp)
            data['sample_embed'] = samples
            data['feature'] = features
            data['y'] = y
            tmp.append(data)

        train_data, test_data = tmp

    return train_data, test_data

def split_random(X: np.ndarray, y: np.ndarray, n_splits: int, **kwargs) -> pd.DataFrame:
    '''
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
    # Split provided data
    cols = ['train/test', 'split_idx', 'X1', 'X2', 'y']
    data = []
    kfold = KFold(n_splits=n_splits)
    for i, (train_idx, test_idx) in enumerate(kfold.split(X)):
        # Append to data for split_guide
        train_rows = list(zip(['train' for _ in range(train_idx.size)], [i for _ in range(train_idx.size)], X[train_idx, 0], X[train_idx, 1], y[train_idx].reshape(-1,)))
        test_rows = list(zip(['test' for _ in range(test_idx.size)], [i for _ in range(test_idx.size)], X[test_idx, 0], X[test_idx, 1], y[test_idx].reshape(-1,)))
        data += train_rows
        data += test_rows

    return pd.DataFrame(data=data, columns=cols)

def split_rcmcs(X: np.ndarray, y: np.ndarray, n_splits: int, cluster_path: Path, idx_feature: dict, **kwargs) -> pd.DataFrame:
    '''
    Splits clusters based on reaction center MCS
    '''
    rxn_id_2_cluster = load_json(cluster_path.with_suffix(".json"))
    cluster_2_rxn_id = defaultdict(list)
    for r, c in rxn_id_2_cluster.items():
        cluster_2_rxn_id[c].append(r)

    rxn_id_2_idx = {v : k for k, v in idx_feature.items()}

    return _split_clusters(X, y, cluster_2_rxn_id, rxn_id_2_idx, n_splits, check_side=1)

def split_homology(X: np.ndarray, y: np.ndarray, n_splits: int, cluster_path: Path, idx_sample: dict, **kwargs) -> pd.DataFrame:
    cluster_id_2_upid = parse_cd_hit_clusters(cluster_path.with_suffix(".clstr"))
    upid_2_idx = {val : key for key, val in idx_sample.items()}

    return _split_clusters(X, y, cluster_id_2_upid, upid_2_idx, n_splits, check_side=0)

def _split_clusters(X: np.ndarray, y: np.ndarray, cluster_2_elt: dict, elt_2_idx: dict, n_splits: int, check_side: int) -> pd.DataFrame:
    '''
    Splits clusters, returns dataframe split guide

    Args
    ----
    X
        Pair datapoints (n_samples x 2)
    y
        Datapoint labels
    cluster_2_elt:dict
        Cluster number -> list of element (protein or reaction) ids
    elt_2_idx:dict
        Element id -> adjacency matrix row / col idx
    check_side:int
        Which half of the pair. 0 = protein, 1 = reaction
    '''
    clusters = np.array(list(cluster_2_elt.keys())).reshape(-1, 1)
    cols = ['train/test', 'split_idx', 'X1', 'X2', 'y']
    data = []
    kfold = KFold(n_splits=n_splits)
    for i, (_, test) in enumerate(kfold.split(clusters)):
        test_clstrs = clusters[test].reshape(-1)
        test_elts = chain(*[cluster_2_elt[cid] for cid in test_clstrs])
        test_elt_idxs = [elt_2_idx[elt] for elt in test_elts]

        for j, pair in enumerate(X):
            if pair[check_side] in test_elt_idxs:
                data.append(['test', i, pair[0], pair[1], y[j]])
            else:
                data.append(['train', i, pair[0], pair[1], y[j]])

    return pd.DataFrame(data=data, columns=cols)

def parse_cd_hit_clusters(filepath: Path):
        '''
        Returns dict of clustered proteins given
        a filepath to a .clstr output file of cd-hit
        '''
        clusters = defaultdict(list)
        current_cluster = None

        with open(filepath, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith(">Cluster"):
                    current_cluster = int(line.split()[1])
                else:
                    clusters[current_cluster].append(line.split()[2][1:-3])

        return clusters