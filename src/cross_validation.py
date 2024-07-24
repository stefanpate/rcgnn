from src.utils import construct_sparse_adj_mat, write_shell_script, load_embed
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
from itertools import product, chain
import json
from collections import namedtuple, defaultdict
import subprocess

BatchScriptParams = namedtuple(typename="BatchScriptParams", field_names=['allocation', 'partition', 'mem', 'time', 'script'])

class BatchGridSearch:
    data_dir = "/projects/p30041/spn1560/hiec/data"
    scratch_dir = "/scratch/spn1560"
    valid_split_strategies = ['random', 'homology']
    past_gs_names = "/home/spn1560/hiec/artifacts/past_grid_search_names.txt"
    embed_dims = {
        'esm': 1280
    }
    
    def __init__(
            self,
            dataset_name:str,
            toc:str,
            neg_multiple:int,
            gs_name:str,
            n_splits:int,
            split_strategy:str,
            embed_type:str,
            seed:int,
            split_sim_threshold:float = 1.0,
            batch_script_params:BatchScriptParams | None = None,
            hps:dict | None = None,
            res_dir:str = "/projects/p30041/spn1560/hiec/artifacts/model_evals/gnn",
            ) -> None:
        
        if split_strategy not in self.valid_split_strategies:
            raise ValueError(f"Invalid split strategy: {split_strategy}. Choose one from: {', '.join(self.valid_split_strategies)}")

        self.dataset_name = dataset_name
        self.toc = toc
        self.neg_multiple = neg_multiple
        self.gs_name = gs_name
        self.n_splits = n_splits
        self.split_strategy = split_strategy
        self.threshold = split_sim_threshold
        self.embed_type = embed_type
        self.embed_dim = self.embed_dims[embed_type]
        self.seed = seed
        self.hps = [{list(hps.keys())[i] : elt[i] for i in range(len(elt))} for elt in product(*hps.values())] if hps is not None else None
        self.rng = np.random.default_rng(seed)
        self.batch_script_params = batch_script_params
        self.adj, self.idx_sample, self.idx_feature = self._get_adjacency_matrix()
        self.X_pos = list(zip(*self.adj.nonzero()))
        self.split_guide_pref = f"{dataset_name}_{toc}_{split_strategy}_threshold_{int(self.threshold * 100)}_{n_splits}_splits_neg_multiple_{neg_multiple}_seed_{seed}"
        self.hp_scratch_pref  = f"{self.scratch_dir}/{gs_name}"
        self.res_dir = f"{res_dir}/{dataset_name}_{toc}_{gs_name}"
        self.gs_params = "\n".join([f"n_splits: {self.n_splits}", f"split_strategy: {self.split_strategy}", f"similarity_threshold: {self.threshold}", f"neg_multiple: {self.neg_multiple}", f"embed_type: {self.embed_type}", f"seed: {self.seed}"])

    def run(self):
        self.setup()

        # Run shell scripts
        for hp_idx, _ in enumerate(self.hps):
            for split_idx in range(self.n_splits):
                arg_str = f"-d {self.dataset_name} -t {self.toc} -a {self.split_strategy} -r {self.threshold} -e {self.seed} -n {self.n_splits} -m {self.neg_multiple} -b {self.embed_type} -s {split_idx} -p {hp_idx} -g {self.gs_name}"
                job_name = f"{self.gs_name}_hps_{hp_idx}_split_{split_idx}"
        
                shell_script = write_shell_script(
                    *self.batch_script_params,
                    arg_str,
                    job_name
                )
                
                with open("batch.sh", 'w') as f:
                    f.write(shell_script)

                subprocess.run(["sbatch", "batch.sh"])

    def setup(self):
        self._check_gs_name()

        if self.hps is None:
            raise NameError("Cannot run grid search without provided hyperparameter dict.")
        
        if self._check_for_split_guide():
            pass
        else:
            X, y = self.sample_negatives()
            _ = self.split_data(X, y)

        for i in range(self.n_splits):
            self.load_data_split(i, setup=True)
        
        self._save_hps_to_scratch()
        self._set_up_res_dir()
    
    def split_data(self, X, y, do_save=True):
        split_guide_path =f"{self.scratch_dir}/{self.split_guide_pref}.csv"
        if self._check_for_split_guide():      
            split_guide = pd.read_csv(split_guide_path, sep='\t')
            return split_guide
        
        if self.threshold < 0.0 or self.threshold > 1.0:
            raise ValueError("Please provide threshold from [0, 1]")
        
        # Shuffle data
        p = self.rng.permutation(X.shape[0])
        X = X[p]
        y = y[p]

        if self.split_strategy == 'random':
            split_guide = self._split_random(X, y)
        elif self.split_strategy == 'homology':
            split_guide = self._split_homology(X, y)
        elif self.split_strategy == 'embedding':
            # TODO: implement _split_embedding
            pass

        if do_save:
            split_guide.to_csv(split_guide_path, sep='\t', index=False)

        return split_guide
            
    def _split_homology(self, X, y):
        cluster_path = f"/home/spn1560/hiec/data/{self.dataset_name}/{self.toc}_{int(self.threshold * 100)}.clstr"
        if not os.path.exists(cluster_path):
            raise ValueError(f"Cluster file does not exist for {self.dataset_name}, {self.toc}, threshold: {self.threshold}")

        cluster_id_2_upid = self._parse_cd_hit_clusters(cluster_path)
        upid_2_idx = {val : key for key, val in self.idx_sample.items()}
        clusters = np.array(list(cluster_id_2_upid.keys())).reshape(-1, 1)

        # Split by cluster
        cols = ['train/test', 'split_idx', 'X1', 'X2', 'y']
        data = []
        kfold = KFold(n_splits=self.n_splits)
        for i, (_, test) in enumerate(kfold.split(clusters)):
            #TODO: Can I do this by only touching test
            test_clstrs = clusters[test].reshape(-1)
            test_up = chain(*[cluster_id_2_upid[cid] for cid in test_clstrs])
            test_prot_idx = [upid_2_idx[upid] for upid in test_up]

            for j, pair in enumerate(X):
                if pair[0] in test_prot_idx:
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
    
    def _save_hps_to_scratch(self):
        for hp_idx, hp in enumerate(self.hps):
            with open(f"{self.hp_scratch_pref}_{hp_idx}_hp_idx.json", 'w') as f:
                json.dump(hp, f)
    
    def load_hps_from_scratch(self, hp_idx):
        with open(f"{self.hp_scratch_pref}_{hp_idx}_hp_idx.json", 'r') as f:
            hp = json.load(f)

        return hp
    
    def _set_up_res_dir(self):
        # Write a hp idx csv
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)

        # Save model hyperparam csv
        cols = self.hps[0].keys()
        data = [list(elt.values()) for elt in self.hps]
        hp_df = pd.DataFrame(data=data, columns=cols)
        hp_df.to_csv(f"{self.res_dir}/hp_toc.csv", sep='\t')

        # Save gridsearch param
        with open(f"{self.res_dir}/gs_params.txt", 'w') as f:
            f.write(self.gs_params)

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
    
    def _check_gs_name(self):
        # Check gs_name not used before
        with open(self.past_gs_names, 'r') as f:
            old_gs_names = [elt.rstrip() for elt in f.readlines()]

        if self.gs_name in old_gs_names:
            raise ValueError(f"{self.gs_name} has already been used as a grid search name")

        old_gs_names.append(self.gs_name) # Add current gs_name

        with open(self.past_gs_names, 'w') as f:
            f.write('\n'.join(elt for elt in old_gs_names))

    def _check_for_split_guide(self):
        split_guide_path =f"{self.scratch_dir}/{self.split_guide_pref}.csv" 
        if os.path.exists(split_guide_path):
            print(f"Found existing data split guide: {self.split_guide_pref}")
            return True
        else:
            return False