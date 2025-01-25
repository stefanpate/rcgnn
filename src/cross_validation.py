from src.utils import load_json
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from collections import defaultdict
from pathlib import Path

def sample_negatives(X: list[tuple[int]], y: list[int], neg_multiple: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    '''
    Samples negatives to bring the ratio of positives to negatives to neg_multiple

    Args
    ----
    X:list[tuple[int]]
        List of (sample, feature) pairs
    y:list[int]
        List of labels
    neg_multiple:int
        Ratio of negatives to positives
    rng:np.random.Generator
        Random number generator

    Returns
    -------
    X:np.ndarray
        Sample, feature pairs
    y:np.ndarray
        Labels

    '''
    if len(X) != len(y):
        raise ValueError("Lengths of X an y must match.")

    # Break up into pos and neg
    X_pos = []
    X_neg = []
    row_idxs = set()
    col_idxs = set()
    for elt, label in zip(X, y):
        row_idxs.add(elt[0])
        col_idxs.add(elt[1])
        if label == 1:
            X_pos.append(elt)
        else:
            X_neg.append(elt)

    row_idxs = list(row_idxs)
    col_idxs = list(col_idxs)
    n_neg_samples = (neg_multiple * len(X_pos)) - len(X_neg)
    
    # Sample subset of unobserved pairs
    while len(X_neg) < n_neg_samples:
        i = rng.choice(row_idxs, size=(1,))[0]
        j = rng.choice(col_idxs, size=(1,))[0]

        if (i, j) not in X_pos and (i, j) not in X_neg:
            X_neg.append((i, j))

    # Concat full dataset
    X = X_pos + X_neg
    y = [1 for _ in range(len(X_pos))] +  [0 for _ in range(len(X_neg))]

    return X, y

def random_split(
        X: np.ndarray,
        y: np.ndarray,
        n_inner_splits: int,
        test_percent: int,
        seed: int,
    ) -> tuple[list, tuple]:
    '''
    Random single outer split followed by kfold inner splits
    '''

    # TODO: orient around seed instead of np.rng for reproducibility across packages
    kfold = KFold(n_splits=n_inner_splits, shuffle=True, random_state=seed)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_percent / 100, shuffle=True, random_state=seed)
    train_val_splits = []
    for _, val_idx in kfold.split(X_train_val):
        X_val = [X_train_val[i] for i in val_idx]
        y_val = [y_train_val[i] for i in val_idx]

        train_val_splits.append((X_val, y_val))

    test = (X_test, y_test)

    return train_val_splits, test

def stratified_sim_split(
        X: np.ndarray,
        y: np.ndarray,
        split_strategy: str,
        split_bounds: list[int],
        n_inner_splits: int,
        test_percent: int,
        cluster_dir: Path,
        adj_mat_idx_to_id: dict,
        dataset: str,
        toc: str,
        rng: np.random.Generator
    ) -> tuple[list, tuple]:
    '''
    Does stratified test + inner kfold splits where clusters are defined by
    levels of similarity.

    Args
    ----
    X:np.ndarray
        Sample, feature pairs expressed as adj mat indices
    y:np.ndarray
        Labels
    split_strategy:str
        'rcmcs' or 'homology'
    split_bounds:list[int]
        List of similarity upper bounds
    n_inner_splits:int
        Number of inner kfold splits
    test_percent:int
        Percentage of data to hold out for testing
    cluster_dir:Path
        Directory containing clustering results
    adj_mat_idx_to_id:dict
        Maps adj mat indices to reaction / protein ids
    dataset:str
        Name of dataset
    toc:str
        Table of contents / what subset of dataset
    rng:np.random.Generator

    Returns
    -------
    train_val_splits:list
        List of kfold splits
    test:tuple
        Test split    
    '''
    def level_split(level_clusters, test_frac, rng, already_sampled = []):
        '''
        Samples clusters for each level l, first removing those clusters w/
        datapoints already sampled at previous levels

        Returns
        -------
        test:list
            List of test indices
        already_sampled:list
            List of datapoint indices already sampled
        '''
        level_test_frac = test_frac / level_clusters.shape[1] # Fraction of l-level test points
        test = []
        for l in range(level_clusters.shape[1]):
            avail_clusters = list(set(level_clusters[:, l]) - set(level_clusters[already_sampled, l]))
            n = int(level_test_frac * level_clusters[:, l].max()) # Number of l-level test clusters / points
            test_clusters = rng.choice(avail_clusters, size=(min(n, len(avail_clusters)),), replace=False)

            for c in test_clusters:
                new = list(np.where(level_clusters[:, l] == c)[0])
                test += new
                already_sampled += new

        return test, already_sampled

    id_to_adj_mat_idx = {v: k for k, v in adj_mat_idx_to_id.items()} # Either for prots or rxns

    # Maps reaction or protein matrix index to pair index in X
    single2pair_idx = defaultdict(list) 
    for i, pair in enumerate(X):
        if split_strategy == 'rcmcs':
            single2pair_idx[pair[1]].append(i) # Orient to rxn matrix idx
        elif split_strategy == 'homology':
            single2pair_idx[pair[0]].append(i) # Orient to prot matrix idx

    # Assemble level clusters matrix
    level_clusters = np.zeros(shape=(len(X), len(split_bounds))) - 1 # (# pairs x # levels of clustering) cols contain jth level cluster idxs
    idxs, cluster_numbers = [], []
    for l, bound in enumerate(split_bounds):
        if bound == 100: # Each id is its own cluster
            point_to_cluster = {id: cid for cid, id in enumerate(id_to_adj_mat_idx.keys())}
        elif split_strategy == 'rcmcs':
            cluster_path = cluster_dir / f"{dataset}_{toc}_{split_strategy}_{bound}.json"
            point_to_cluster = load_json(cluster_path)
        elif split_strategy == 'homology':
            cluster_path = cluster_dir / f"{dataset}_{toc}_{split_strategy}_{bound}.clstr"
            clusters = parse_cd_hit_clusters(cluster_path)
            point_to_cluster = {id: cid for cid, ids in clusters.items() for id in ids}

        for id, cid in point_to_cluster.items():
            for idx in single2pair_idx[id_to_adj_mat_idx[id]]:
                idxs.append(idx)
                cluster_numbers.append(cid)

        level_clusters[idxs, l] = cluster_numbers

    test, already_sampled = level_split(level_clusters, test_percent / 100, rng) # Split train_val / test

    # Split train_val into k folds
    train_val_splits = []
    for i in range(n_inner_splits - 1):
        val, already_sampled = level_split(level_clusters, 1 / n_inner_splits, rng, already_sampled=already_sampled)
        train_val_splits.append(val)

    last_train_val = [i for i in range(level_clusters.shape[0]) if i not in already_sampled]
    train_val_splits.append(last_train_val)

    # Convert to pairs of (prot, rxn) adj matrix indices and y labels
    indices = train_val_splits + [test]
    labeled_pairs = []
    for elt in indices:
        Xi = []
        yi = []
        for idx in elt:
            Xi.append((X[idx]))
            yi.append(y[idx])
        labeled_pairs.append((tuple(Xi), tuple(yi)))

    train_val_splits = labeled_pairs[:-1]
    test = labeled_pairs[-1]
    
    return train_val_splits, test

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

if __name__ == '__main__':
    from src.utils import construct_sparse_adj_mat
    rng = np.random.default_rng(seed=1234)
    dataset = 'sprhea'
    toc = 'v3_folded_test'

    adj, idx_sample, idx_feature = construct_sparse_adj_mat(
        Path('/home/stef/quest_data/hiec/data') / dataset / (toc + ".csv")
    )

    X_pos = list(zip(*adj.nonzero()))

    # train_val, test = stratified_sim_split(
    #     X=X_pos,
    #     y=[1 for _ in range(len(X_pos))],
    #     split_strategy='rcmcs',
    #     split_bounds=[100, 80, 60, 40],
    #     n_inner_splits=3,
    #     test_percent=20,
    #     cluster_dir=Path('/home/stef/hiec/artifacts/clustering'),
    #     adj_mat_idx_to_id=idx_feature,
    #     dataset=dataset,
    #     toc=toc,
    #     rng=rng
    # )

    train_val, test = stratified_sim_split(
        X=X_pos,
        y=[1 for _ in range(len(X_pos))],
        split_strategy='homology',
        split_bounds=[100, 80, 60, 40],
        n_inner_splits=3,
        test_percent=20,
        cluster_dir=Path('/home/stef/hiec/artifacts/clustering'),
        adj_mat_idx_to_id=idx_sample,
        dataset=dataset,
        toc=toc,
        rng=rng
    )

    # random_split(
    #     X_pos,
    #     y=[1 for _ in range(len(X_pos))],
    #     n_inner_splits=3,
    #     test_percent=20,

    # )