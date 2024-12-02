import hydra
from omegaconf import DictConfig
from functools import partial
from itertools import product
from src.utils import construct_sparse_adj_mat
from src.cross_validation import load_data_split
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from pathlib import Path
import torch

def _sample_negatives(X_pos: list[tuple[int]], neg_multiple: int, rng: np.random.Generator):
    '''
    Samples neg_multiple * len(X_pos) negative pairs from the adjacency matrix implied by X_pos
    '''
    n_pos_samples = len(X_pos)
    n_rows, n_cols = [max(elt) for elt in list(zip(*X_pos))]
    n_neg_samples = neg_multiple * n_pos_samples
    
    # Sample subset of unobserved pairs
    X_neg = []
    while len(X_neg) < n_neg_samples:
        i = rng.integers(0, n_rows)
        j = rng.integers(0, n_cols)

        if (i, j) not in X_pos and (i, j) not in X_neg:
            X_neg.append((i, j))

    # Concat full dataset
    X = np.vstack(X_pos + X_neg)
    y = np.hstack([np.ones(shape=(len(X_pos,))), np.zeros(shape=(len(X_neg,)))])

    return X, y

def _split_data(X: np.ndarray, y: np.ndarray, splitter: callable, rng: np.random.Generator):
    # Shuffle data
    p = rng.permutation(X.shape[0])
    X = X[p]
    y = y[p]

    split_guide = splitter(X, y)
    
    return split_guide


def _split_random(X: np.ndarray, y: np.ndarray, n_splits: int, **kwargs) -> pd.DataFrame:
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

def _balance_test_set(split_guide: pd.DataFrame, rng: np.random.Generator):
        to_remove = []
        n_splits = split_guide['split_idx'].max() + 1
        for i in range(n_splits):
            sel_ds = (split_guide['split_idx'] == i) & (split_guide['train/test'] == 'test') & (split_guide['y'] == 0)
            sel_other = (split_guide['split_idx'] == i) & (split_guide['train/test'] == 'test') & (split_guide['y'] == 1)
            other = split_guide[sel_other]
            this = split_guide[sel_ds]
            indices = this.index
            rm = rng.choice(indices, size=len(this) - len(other), replace=False)
            to_remove.append(rm)
        
        to_remove = np.hstack(to_remove)
        split_guide.drop(labels=to_remove, inplace=True)

def make_split_guide(X_pos: np.ndarray, neg_multiple: int, splitter: callable, rng: np.random.Generator):
    X, y = _sample_negatives(X_pos, neg_multiple, rng)
    split_guide = _split_data(X, y, splitter, rng)

    if neg_multiple > 1:
        _balance_test_set(split_guide, rng)

    return split_guide

# TODO: change to caching smarts and rcs to avoid loading reaction dataset unneccesarily?
def cache_data(scratch_path: Path, data_path: Path, split_guide: pd.DataFrame, dataset: str, idx_sample: dict = {}, idx_feature: dict = {}):
    n_splits = split_guide['split_idx'].max() + 1
    splits = [i for i in range(n_splits)]
    suffs = ['train', 'test']

    if scratch_path.exists() and all([(scratch_path / f"{i}_{s}.npy").exists() for i, s in product(splits, suffs)]):
        return
    
    scratch_path.mkdir(parents=True)

    for s in splits:
        train_data, test_data = load_data_split(
            split_idx=s,
            dataset=dataset,
            data_path=data_path,
            idx_sample=idx_sample,
            idx_feature=idx_feature,
            split_guide=split_guide
        )
        
        # Cache
        np.save(scratch_path / f"{s}_train.npy", train_data)
        np.save(scratch_path / f"{s}_test.npy", test_data)

def load_embed(path: Path, embed_key: int):
    id = path.stem
    f = torch.load(path)
    if type(f) == dict:
        embed = f['mean_representations'][embed_key]
    elif type(f) == torch.Tensor:
        embed = f
    return id, embed

splitters = {
    'random': _split_random,

}

@hydra.main(version_base=None, config_path="../configs", config_name="prep_data")
def main(cfg: DictConfig):
    rng = np.random.default_rng(seed=cfg.data.seed)
    splitter = partial(
        splitters[cfg.data.split_strategy],
            split_bound=cfg.data.split_bound / 100, # Bound stored as percent
            n_splits=cfg.data.n_splits
    )

    adj, idx_sample, idx_feature = construct_sparse_adj_mat(
        Path(cfg.filepaths.data) / cfg.data.dataset / (cfg.data.toc + ".csv")
    )
    X_pos = list(zip(*adj.nonzero()))

    split_guide = make_split_guide(
        X_pos=X_pos,
        neg_multiple=cfg.data.neg_multiple,
        splitter=splitter,
        rng=rng
    )

    split_guide.to_parquet('split_guide.parquet')

    cache_data(
        scratch_path=Path(cfg.filepaths.scratch) / cfg.data.subdir_patt,
        data_path=Path(cfg.filepaths.data),
        split_guide=split_guide,
        dataset=cfg.data.dataset,
        idx_sample=idx_sample,
        idx_feature=idx_feature
    )

if __name__ == '__main__':
    main()