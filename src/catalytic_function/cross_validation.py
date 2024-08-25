'''
TODO
Set up internal run 
Toggle on/off when to run with option "resume" defaults to flase
Call bgs.run(hps) for all returned objects outside of load_from_exp

'''

from catalytic_function.utils import construct_sparse_adj_mat, load_embed, load_json
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
from itertools import product, chain
import json
from collections import defaultdict
import subprocess
from dataclasses import dataclass, asdict, fields

DEFAULT_DATA_DIR = "/projects/b1039/trw7841/hiec/data"
DEFAULT_SCRATCH_DIR = "/scratch/trw7841"

@dataclass
class BatchScript:
    '''
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
    allocation:str
    partition:str
    mem:str
    time:str
    script:str

    def write(self, arg_str, job_name):
        cpu_blacklist = ["#SBATCH --gres=gpu:a100:1"]
        lines = [
            f"#!/bin/bash",
            f"#SBATCH -A {self.allocation}",
            f"#SBATCH -p {self.partition}",
            f"#SBATCH --gres=gpu:a100:1",
            f"#SBATCH -N 1",
            f"#SBATCH -n 1",
            f"#SBATCH --mem={self.mem}",
            f"#SBATCH -t {self.time}:00:00",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output=../logs/out/{job_name}",
            f"#SBATCH --error=../logs/error/{job_name}",
            f"#SBATCH --mail-type=END",
            f"#SBATCH --mail-type=FAIL",
            f"#SBATCH --mail-user=pasha@u.northwestern.edu",
            f"ulimit -c 0",
            # f"module load python/anaconda3.6",
            # f"module load gcc/9.2.0",
            # f"source activate hiec",
            f"pixi run --frozen python -u {self.script} {arg_str}",
        ]

        if self.partition == 'gengpu':
            return '\n'.join(lines)
        else:
            return '\n'.join([line for line in lines if line not in cpu_blacklist])


@dataclass
class HyperHyperParams:
    dataset_name:str
    toc:str
    neg_multiple:str
    n_splits:int
    split_strategy:str
    embed_type:str
    seed:int
    split_sim_threshold:float = 1.0
    embed_dim:int = None

    def __post_init__(self):
        valid_split_strategies = ['random', 'homology', 'rcmcs']
        if self.split_strategy not in valid_split_strategies:
            raise ValueError(f"Invalid split strategy: {self.split_strategy}. Choose one from: {', '.join(valid_split_strategies)}")
        
        embed_dims = {
            'esm': 1280,
        }
        
        if not self.embed_dim:
            try:
                self.embed_dim = embed_dims[self.embed_type]
            except KeyError:
                raise KeyError(f"Embed type {self.embed_type} unknown. Select from {list(embed_dims.keys())}")

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_single_experiment(cls, single_experiment:dict):
        field_names = [field.name for field in fields(cls)]
        hhp_args = {k : v for k,v in single_experiment.items() if k in field_names}
        return cls(**hhp_args)

def load_single_experiment(hp_idx, scratch_dir=DEFAULT_SCRATCH_DIR):
    with open(f"{scratch_dir}/{hp_idx}_hp_idx.json", 'r') as f:
        hp = json.load(f)

    return hp

class BatchGridSearch:

    def __init__(
            self,
            hhps:HyperHyperParams,
            res_dir:str,
            scratch_dir=DEFAULT_SCRATCH_DIR,
            data_dir=DEFAULT_DATA_DIR,
            ) -> None:

        for k, v in hhps.to_dict().items():
            setattr(self, k, v)

        self.hhps = hhps.to_dict()
        self.res_dir = res_dir
        self.scratch_dir = scratch_dir
        self.data_dir = data_dir
        self.rng = np.random.default_rng(self.seed)
        self.split_guide_pref = f"{self.dataset_name}_{self.toc}_{self.split_strategy}"\
            f"_threshold_{int(self.split_sim_threshold * 100)}_{self.n_splits}_splits"\
            f"_neg_multiple_{self.neg_multiple}_seed_{self.seed}"
        self.experiments = pd.read_csv(f"{res_dir}/experiments.csv", sep='\t', index_col=0)
        self.next_hp_idx = self.experiments.index.max() + 1
        self.adj, self.idx_sample, self.idx_feature = self._get_adjacency_matrix()
        self.X_pos = list(zip(*self.adj.nonzero()))

    def run(self, hps, batch_script):
        if type(hps) is dict:
            hps = [{k : elt[i] for i, k in enumerate(hps.keys())} for elt in product(*hps.values())]

        self._run(hps, batch_script)

    def resume(self, hps, batch_script, chkpt_idxs):
        self._run(hps, batch_script, chkpt_idxs)
    
    def _run(self, hps, batch_script, chkpt_idxs=None):
        hps = [{**elt, **self.hhps} for elt in hps]

        self._setup(hps)

        # Run shell scripts
        for i, _ in enumerate(hps):
            hp_idx = self.next_hp_idx + i
            for split_idx in range(self.n_splits):

                if chkpt_idxs:
                    arg_str = f"-s {split_idx} -p {hp_idx} -c {chkpt_idxs[i]}"
                else:
                    arg_str = f"-s {split_idx} -p {hp_idx}"
                
                job_name = f"cv_hp_idx_{hp_idx}_split_{split_idx}"
                shell_script = batch_script.write(arg_str, job_name)
                
                with open("batch.sh", 'w') as f:
                    f.write(shell_script)

                subprocess.run(["sbatch", "batch.sh"])

    def _setup(self, hps):
        
        if self._check_for_split_guide():
            pass
        else:
            X, y = self.sample_negatives()
            _ = self.split_data(X, y)

        for i in range(self.n_splits):
            self.load_data_split(i, setup=True)
        
        self._save_hps_to_scratch(hps)
        self._append_experiments(hps)
    
    def split_data(self, X, y, do_save=True):
        split_guide_path =f"{self.scratch_dir}/{self.split_guide_pref}.csv"
        
        if self._check_for_split_guide():      
            split_guide = pd.read_csv(split_guide_path, sep='\t')
            return split_guide
        
        if self.split_sim_threshold < 0.0 or self.split_sim_threshold > 1.0:
            raise ValueError("Please provide threshold from [0, 1]")
        
        # Shuffle data
        p = self.rng.permutation(X.shape[0])
        X = X[p]
        y = y[p]

        if self.split_strategy == 'random':
            split_guide = self._split_random(X, y)
        elif self.split_strategy == 'homology':
            split_guide = self._split_homology(X, y)
        elif self.split_strategy == 'rcmcs':
            split_guide = self._split_rcmcs(X, y)

        if do_save:
            split_guide.to_csv(split_guide_path, sep='\t', index=False)

        return split_guide
            
    def _split_homology(self, X, y):
        cluster_path = f"/home/trw7841/src/catalytic-function/data/{self.dataset_name}/{self.toc}_{int(self.split_sim_threshold * 100)}.clstr"
        
        if not os.path.exists(cluster_path):
            raise ValueError(f"Cluster file does not exist for {self.dataset_name}, {self.toc}, strategy: {self.split_strategy} threshold: {self.split_sim_threshold}")

        cluster_id_2_upid = self._parse_cd_hit_clusters(cluster_path)
        upid_2_idx = {val : key for key, val in self.idx_sample.items()}
        clusters = np.array(list(cluster_id_2_upid.keys())).reshape(-1, 1)

        # Split by cluster
        cols = ['train/test', 'split_idx', 'X1', 'X2', 'y']
        data = []
        kfold = KFold(n_splits=self.n_splits)
        for i, (_, test) in enumerate(kfold.split(clusters)):
            test_clstrs = clusters[test].reshape(-1)
            test_up = chain(*[cluster_id_2_upid[cid] for cid in test_clstrs])
            test_prot_idx = [upid_2_idx[upid] for upid in test_up]

            for j, pair in enumerate(X):
                if pair[0] in test_prot_idx:
                    data.append(['test', i, pair[0], pair[1], y[j]])
                else:
                    data.append(['train', i, pair[0], pair[1], y[j]])

        return pd.DataFrame(data=data, columns=cols)
    
    def _split_rcmcs(self, X, y):
        '''
        Splits clusters based on reaction center MCS
        '''
        cluster_path = f"/home/spn1560/hiec/artifacts/clustering/{self.dataset_name}_{self.toc}_{self.split_strategy}_{int(self.split_sim_threshold * 100)}.json"

        if not os.path.exists(cluster_path):
            raise ValueError(f"Cluster file does not exist for {self.dataset_name}, {self.toc}, strategy: {self.split_strategy} threshold: {self.split_sim_threshold}")
        
        rxn_id_2_cluster = load_json(cluster_path)
        cluster_2_rxn_id = defaultdict(list)
        for r, c in rxn_id_2_cluster.items():
            cluster_2_rxn_id[c].append(r)

        rxn_id_2_idx = {v : k for k, v in self.idx_feature.items()}

        return self._split_clusters(X, y, cluster_2_rxn_id, rxn_id_2_idx, 1)

    def _split_clusters(self, X, y, cluster_2_elt, elt_2_idx, check_side):
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
        kfold = KFold(n_splits=self.n_splits)
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

    def _split_random(self, X, y):
        '''
        Args
        ----
        X:list or array - All (sample, feature) index pairs of whole dataset
        y:list or array -  Labels of pairs in X

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
        kfold = KFold(n_splits=self.n_splits)
        for i, (train_idx, test_idx) in enumerate(kfold.split(X)):
            # Append to data for split_guide
            train_rows = list(zip(['train' for _ in range(train_idx.size)], [i for _ in range(train_idx.size)], X[train_idx, 0], X[train_idx, 1], y[train_idx].reshape(-1,)))
            test_rows = list(zip(['test' for _ in range(test_idx.size)], [i for _ in range(test_idx.size)], X[test_idx, 0], X[test_idx, 1], y[test_idx].reshape(-1,)))
            data += train_rows
            data += test_rows

        return pd.DataFrame(data=data, columns=cols)
    
    def sample_negatives(self):
        '''
        Samples n_samples negative pairs from an n_rows x n_cols adjacency matrix
        given obs_pairs, positive pairs which should not be sampled
        '''
        n_pos_samples = len(self.X_pos)
        n_rows, n_cols = [max(elt) for elt in list(zip(*self.X_pos))]
        n_neg_samples = self.neg_multiple * n_pos_samples
        
        # Sample subset of unobserved pairs
        X_neg = []
        while len(X_neg) < n_neg_samples:
            i = self.rng.integers(0, n_rows)
            j = self.rng.integers(0, n_cols)

            if (i, j) not in self.X_pos and (i, j) not in X_neg:
                X_neg.append((i, j))

        # Concat full dataset
        X = np.vstack(self.X_pos + X_neg)
        y = np.hstack([np.ones(shape=(len(self.X_pos,))), np.zeros(shape=(len(X_neg,)))])

        return X, y
    
    def load_data_split(self, split_idx, setup=False):
        '''
        Args
        ----
        embed_type:str
        split_idx:int
        embed_dim:int
        setup:bool - Whether or not to load embeds found in scratch

        Returns
        -------
        (train_data, test_data) | (None, None)
        '''
        fns = [f"{self.split_guide_pref}_{split_idx}_split_idx_{self.embed_type}" + suff for suff in ['_train.npy', '_test.npy']]
        found = all([os.path.exists(f"{self.scratch_dir}/{fn}") for fn in fns])
                
        if found and setup:
            print(f"Found existing data splits for {self.split_guide_pref}, {split_idx}")
            return None, None
        elif found:
            train_data, test_data = [np.load(f"{self.scratch_dir}/{fn}") for fn in fns]
        else:
            split_guide = pd.read_csv(f"{self.scratch_dir}/{self.split_guide_pref}.csv", sep='\t')
            train_split =  split_guide.loc[(split_guide['train/test'] == 'train') & (split_guide['split_idx'] == split_idx)]
            test_split =  split_guide.loc[(split_guide['train/test'] == 'test') & (split_guide['split_idx'] == split_idx)]

            tp = np.dtype([('sample_embed', np.float32, (self.embed_dim,)), ('feature', '<U100'), ('y', int)])
            tmp = []
            for split in [train_split, test_split]:
                samples = []
                features = []
                y = [elt for elt in split.loc[:, 'y']]
                for sample_idx in split.loc[:, 'X1']:
                    sample_name = self.idx_sample[sample_idx]
                    samples.append(load_embed(f"{self.data_dir}/{self.dataset_name}/{self.embed_type}/{sample_name}.pt", embed_key=33)[1])

                for feature_idx in split.loc[:, 'X2']:
                    features.append(self.idx_feature[feature_idx])

                data = np.zeros(len(samples), dtype=tp)
                data['sample_embed'] = samples
                data['feature'] = features
                data['y'] = y
                tmp.append(data)

            train_data, test_data = tmp
            
            # Save for next time
            for (fn, data) in zip(fns, tmp):
                np.save(f"{self.scratch_dir}/{fn}", data)

        return train_data, test_data
    
    def _get_adjacency_matrix(self):
        return construct_sparse_adj_mat(self.dataset_name, self.toc)
    
    def _save_hps_to_scratch(self, hps):
        for i, hp in enumerate(hps):
            hp_idx = self.next_hp_idx + i
            with open(f"{self.scratch_dir}/{hp_idx}_hp_idx.json", 'w') as f:
                json.dump(hp, f)
    
    def _append_experiments(self, hps):
        df = pd.DataFrame(hps)
        df.index += self.next_hp_idx
        new_exp = pd.concat((self.experiments, df))
        new_exp.to_csv(f"{self.res_dir}/experiments.csv", sep='\t')

    def _parse_cd_hit_clusters(self, file_path):
        clusters = defaultdict(list)
        current_cluster = None

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith(">Cluster"):
                    current_cluster = int(line.split()[1])
                else:
                    clusters[current_cluster].append(line.split()[2][1:-3])

        return clusters

    def _check_for_split_guide(self):
        split_guide_path =f"{self.scratch_dir}/{self.split_guide_pref}.csv" 
        if os.path.exists(split_guide_path):
            print(f"Found existing data split guide: {self.split_guide_pref}")
            return True
        else:
            return False

if __name__ == '__main__':
    dataset_name = 'sprhea'
    toc = 'v3_folded_pt_ns' # Name of file with protein id | features/labels | sequence
    n_splits = 3
    seed = 1234
    neg_multiple = 1
    split_strategy = 'rcmcs'
    split_sim_threshold = 0.8
    embed_type = 'esm'
    res_dir = "/projects/p30041/spn1560/hiec/artifacts/model_evals/gnn"

    # Configurtion stuff
    hhps = HyperHyperParams(
        dataset_name=dataset_name,
        toc=toc,
        neg_multiple=neg_multiple,
        n_splits=n_splits,
        split_strategy=split_strategy,
        embed_type=embed_type,
        seed=seed,
        split_sim_threshold=split_sim_threshold,
    )

    # Create grid search object
    gs = BatchGridSearch(
        hhps=hhps,
        res_dir=res_dir,
    )

    X, y = gs.sample_negatives()
    split_guide = gs.split_data(X, y)

    gb = split_guide.groupby(by='split_idx')
    pos_neg = []
    for name, group in gb:
        train = group.loc[group["train/test"] == "train", ["X1", "X2"]].values.tolist()
        train = set([tuple(elt) for elt in train])
        test = group.loc[group["train/test"] == "test", ["X1", "X2"]].values.tolist()
        test = set([tuple(elt) for elt in test])

        assert len(train & test) == 0

        train_labels = list(chain(*group.loc[group["train/test"] == "train", ["y"]].values.tolist()))
        test_labels = list(chain(*group.loc[group["train/test"] == "test", ["y"]].values.tolist()))

        train_pos_neg = (len(train_labels) - sum(train_labels)) / len(train_labels)
        test_pos_neg = (len(test_labels) - sum(test_labels)) / len(test_labels)
        pos_neg.append({"train": train_pos_neg, "test": test_pos_neg})

    print(pos_neg)

    print('hold')
