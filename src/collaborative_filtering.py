import os
import pandas as pd
import subprocess
import numpy as np
from src.utils import load_embed, save_json, load_json
import scipy as sp
from itertools import product
from sklearn.metrics import precision_recall_curve
import time
from collections import defaultdict
from datetime import datetime

class cf:
    
    def __init__(self, X_name:str, Y_name:str, embed_type, master_feature_idxs:dict,
                feature_feature_sim_mats=[], batch_size=15000
                ):
        self.X_name = X_name
        self.Y_name = Y_name
        self.ffsm = feature_feature_sim_mats
        self.embed_type = embed_type
        self.idxs = {}
        self.idxs['feature'] = master_feature_idxs
        self.X_idxs = None
        self.Y_idxs = None
        self.batch_size = batch_size
        self.shapes = {}
        self.knn_thresholds = None
        self.timestamp = datetime.now().strftime("%y%m%d_%H_%M_%S")

        # Get id to index for samples and features
        nf = len(self.idxs['feature'])
        self.idxs[self.X_name], nk = self._read_toc(self.X_name, return_idxs=True) # nk number known samples
        self.idxs[self.Y_name], nu = self._read_toc(self.Y_name, return_idxs=True) # nu number of unknown samples

        self.shapes[self.X_name] = (nk, nf)
        self.shapes[self.Y_name] = (nu, nf)

        # TODO: run read_toc for all ffsm

        self.n_batches = nk // self.batch_size + 1 # Batch along nk >> nu


    def _read_toc(self, ds_name, return_idxs=False):
        '''
        Read table of contents csv for dataset.
        Store names/ids in dict.
        '''
        path = f"../data/{ds_name}/{ds_name}.csv"
        df = pd.read_csv(path, delimiter='\t')
        df.set_index('Entry', inplace=True)

        if return_idxs:
            sample_idxs = {k: i for i, k in enumerate(df.index)}
            n_samples = len(sample_idxs)
            return sample_idxs, n_samples
        else:
            return df

    
    def batch_construct_dense_sim_mats(self): 
        # Check for existing sim_mats
        print("Checking for similarity matrices")
        path_pref = self.get_sim_mat_path_pref(self.embed_type, self.X_name, self.Y_name)
        sim_mat_dir = '/'.join(path_pref.split('/')[:-1])
        fn_pref = path_pref.split('/')[-1]
        matches = [elt for elt in os.listdir(sim_mat_dir) if fn_pref in elt]

        
        # Construct
        if len(matches) != self.n_batches:
            print("Constructing similarity matrices")
            for elt in matches:
                subprocess.run(["rm", f"{sim_mat_dir}/{elt}"])
            
            # Load embeds
            left_embeds = self.load_dense_embeds(self.X_name, self.embed_type)

            # Avoid loading same thing twice
            if self.X_name == self.Y_name:
                right_embeds = left_embeds
            else:
                right_embeds = self.load_dense_embeds(self.Y_name, self.embed_type)

            # Matmul and save sim_batch
            print("Saving similarity matrices")
            for i in range(self.n_batches):
                path_pref = self.get_sim_mat_path_pref(self.embed_type, self.X_name, self.Y_name)
                sim_batch = left_embeds[i * self.batch_size : (i + 1) * self.batch_size] @ right_embeds.T
                np.save(path_pref + f"_batch_{i}.npy", sim_batch)
            
    def get_sparse_adj_mat(self, ds_name):
        path = self.get_adj_mat_path(ds_name)
        row_idxs = self.idxs[ds_name]
        col_idxs = self.idxs['feature']
        shape = (len(row_idxs), len(col_idxs))
        if os.path.exists(path):
            print(f"Loading {ds_name} sparse adjacency matrix")
            adj = sp.sparse.load_npz(path)

        else:
            # Construct ground truth protein-function matrix
            print(f"Constructing {ds_name} sparse adjacency matrix")
            df = self._read_toc(ds_name)
            row, col, data = [], [], [] # For csr
            for i, elt in enumerate(row_idxs):
                ecs = df.loc[elt, 'EC number'].split(';')
                i = row_idxs[elt]
                for ec in ecs:
                    j = self.idxs['feature'][ec]
                    row.append(i)
                    col.append(j)
                    data.append(1)
                
                print(f"{i}/{shape[0]}", end='\r')

            adj = sp.sparse.csr_array((data, (row, col)), shape=shape)
            print("\nSaving sparse adjacency matrix")
            sp.sparse.save_npz(path, adj)

        return adj

    def _kfold_splits(self, kfold, seed):
        rng = np.random.default_rng(seed=seed)
        cv_idxs = np.arange(self.shapes[self.X_name][0])
        fold_size = int(self.shapes[self.X_name][0] / kfold)
        rng.shuffle(cv_idxs)
        splits = [cv_idxs[i * fold_size : (i + 1) * fold_size] for i in range(kfold)]
        return splits
    
    def _precompute_knn_thresholds(self, ks, splits, kfold=1):
        '''
        Pre-computes k-thresholds for multiple ks for later
        kfold cv / HPO. Zeros out rows & cols in sim mat corresponding
        to the test split.
        '''
        print("kNN thresholding & normalizing")
        path_pref = self.get_sim_mat_path_pref(self.embed_type, self.X_name, self.Y_name)
        max_k = max(ks) # k-thresholds for lesser ks come along with max_k
        for i in range(self.n_batches):

            # Load batch sim mat
            path = path_pref + f"_batch_{i}.npy"
            batch_sim_mat = np.load(path)

            if i == 0:
                d = batch_sim_mat.shape[1] # Infer last dimension from 1st batch
                max_threshes = np.zeros(shape=(kfold, max_k, d)) # Create 3d array to store n_splits (max_k x d) "top-k" matrices             

            for j, split in enumerate(splits):

                    # Zero out rows & cols at in case of HPO (kfold > 1)
                    batch_sim_mat, row_vals, col_vals = self._mask_at_idxs(batch_sim_mat, split, batch_no=i)

                    # Update this split's max_threshes top-k matrix
                    max_threshes[j] = np.sort(np.vstack((max_threshes[j], batch_sim_mat)), axis=0)[-max_k:, :]

                    # Put back masked values
                    batch_sim_mat = self._unmask_at_idxs(batch_sim_mat, split, row_vals, col_vals, batch_no=i)

        # Store precomputed k-threshes in (kfold x d) matrices under key (k(NN), k(fold))
        self.knn_thresholds = {}
        for k in ks:
            self.knn_thresholds[k, kfold] = max_threshes[:, -k, :]

    def _batch_knn_thresholds(self, k):
        print("kNN thresholding & normalizing")
        path_pref = self.get_sim_mat_path_pref(self.embed_type, self.X_name, self.Y_name)
        for i in range(self.n_batches):

            path = path_pref + f"_batch_{i}.npy"
            sim_mat_i = np.load(path)

            # Infer shape from first batch
            if i == 0:
                threshes = np.zeros(shape=(k, sim_mat_i.shape[1]))
            
            threshes_i = np.sort(sim_mat_i, axis=0)[-k:, :]
            threshes = np.sort(np.vstack((threshes, threshes_i)), axis=0)[-k:, :]

        threshes = threshes[-k, :].reshape(1,-1)

        return threshes
    
    # def _zero_at_idxs(self, mat, idxs, batch_no=None):
    #     if batch_no is not None:
    #         lb, ub = batch_no * self.batch_size, (batch_no + 1) * self.batch_size
    #         idxs = idxs[(idxs >= lb) & (idxs < ub)]
    #         idxs = idxs % self.batch_size

    #     mat[idxs, :] = 0
    #     mat[:, idxs] = 0

    #     return mat

    def _mask_at_idxs(self, mat, idxs, batch_no=None):
        if batch_no is not None:
            lb, ub = batch_no * self.batch_size, (batch_no + 1) * self.batch_size
            idxs = idxs[(idxs >= lb) & (idxs < ub)]
            idxs = idxs % self.batch_size

        row_vals = mat[idxs, :]
        col_vals = mat[:, idxs]

        mat[idxs, :] = 0
        mat[:, idxs] = 0

        return mat, row_vals, col_vals
    
    def _unmask_at_idxs(self, mat, idxs, row_vals, col_vals, batch_no=None):
        if batch_no is not None:
            lb, ub = batch_no * self.batch_size, (batch_no + 1) * self.batch_size
            idxs = idxs[(idxs >= lb) & (idxs < ub)]
            idxs = idxs % self.batch_size

        mat[idxs, :] = row_vals
        mat[:, idxs] = col_vals

        return mat

    def fit(self):
        self.batch_construct_dense_sim_mats()
        self.X = self.get_sparse_adj_mat(self.X_name)
        self.Y = self.get_sparse_adj_mat(self.Y_name)

    
    def predict(self, k, cv_idxs=None, kfold=1, split_no=0):
        # TODO: Bring in feature side / ensembling
        path_pref = self.get_sim_mat_path_pref(self.embed_type, self.X_name, self.Y_name)
        left = self.X
        right_hat = sp.sparse.csr_array(self.shapes[self.Y_name]) # Init empty sparse arr

        # Get k-thresholds
        if self.knn_thresholds is None:
            k_thresholds = self._batch_knn_thresholds(k) # Predicting outside train set
        else:
            k_thresholds = self.knn_thresholds[k, kfold][split_no] # Cross validation
        
        print("Predicting")
        for i in range(self.n_batches):
            path = path_pref + f"_batch_{i}.npy"
            sim_mat_i = np.load(path)

            # Zero out rows & cols at cv_idxs (e.g., HPO)
            if cv_idxs is not None:
                sim_mat_i, _, _ = self._mask_at_idxs(sim_mat_i, cv_idxs, batch_no=i)

            # Infer shape from first batch
            if i == 0:
                sums = np.zeros(shape=(1, sim_mat_i.shape[1]))

            sim_mat_i[sim_mat_i < k_thresholds] = 0 # k-threshold
            sums += sim_mat_i.sum(axis=0).reshape(1, -1) # Update sums after k-threshold
            
            right_hat += sp.sparse.csr_array(left.T[:, i * self.batch_size : (i + 1) * self.batch_size] @ sim_mat_i).T

        # Normalize
        right_hat = right_hat.toarray()
        sums = sums.T
        where_not_zero = (sums != 0).reshape(-1,)
        right_hat[where_not_zero] /= sums[where_not_zero]

        return right_hat

    def _evaluate(self, Y_hat, test, cv_idxs=None):
        print("Evalutating")
        Y_true = self.Y.toarray()

        # Pick out test set if HPO
        if cv_idxs is not None:
            Y_true = Y_true[cv_idxs, :].ravel()
            Y_hat = Y_hat[cv_idxs, :].ravel()

        metric = test(Y_true, Y_hat)
        return metric
    
    def kfold_knn_opt(self, kfold, grid_search:dict, seed=1234):
        splits = self._kfold_splits(kfold, seed)
        ks = grid_search['ks']
        gs_keys = list(grid_search.keys())
        gs_values = list(grid_search.values())
        hyperparams = list(product(*gs_values))

        self._precompute_knn_thresholds(ks, splits, kfold)

        res = defaultdict(list)
        # TODO: use gs vals and keys to do multiple hypparams
        for j, elt in enumerate(hyperparams):

            # Enter next tuple of hp values
            for i in range(len(elt)):
                res[gs_keys[i]].append(elt[i])
            
            inside_F1s = []
            inside_thresholds = []
            for s, split in enumerate(splits):
                Y_hat = self.predict(elt[0], cv_idxs=split, kfold=kfold, split_no=s)
                metric = self._evaluate(Y_hat, precision_recall_curve, split)
                precision, recall, thresholds_pr = metric
                F1 = (2 * precision * recall) / (precision + recall) # Harmonic mean
                best_F1 = np.max(F1)
                best_threshold = thresholds_pr[np.argmax(F1)]
                inside_F1s.append(best_F1)
                inside_thresholds.append(best_threshold)

            # Enter performance metrics
            inside_thresholds, inside_F1s = np.array(inside_thresholds), np.array(inside_F1s)
            res["F1"].append(inside_F1s.mean())
            res["F1_err"].append(inside_F1s.std() / np.sqrt(kfold))
            res["thresholds"].append(inside_thresholds.mean())
            res["thresholds_err"].append(inside_thresholds.std() / np.sqrt(kfold))

            # Save
            save_json(res, f"../artifacts/cf/{kfold}_fold_hpo_{self.X_name}_{self.embed_type}_{self.timestamp}.json")
            print(f"Last hyperparameter set saved: {gs_keys} = {elt}, # {j+1} / {len(hyperparams)}")

        return res
    
    def get_adj_mat_path(self, ds_name):
        return f"../data/{ds_name}/cf_adj_mat.npz"

    def get_sim_mat_path_pref(self, type, left, right):
        return f"/scratch/spn1560/{type}_{left}_{right}"
    
    def load_dense_embeds(self, ds_name, embed_type, do_norm=True):
        print("Loading dense embeddings")
        magic_key = 33
        data_path = f"../data/{ds_name}/"
        ds_idxs = self.idxs[ds_name]
        embeds = []
        for i, elt in enumerate(ds_idxs):
            embeds.append(load_embed(data_path + f"{embed_type}/{elt}.pt", embed_key=magic_key)[1])

            if i % 5000 == 0:
                print(f"Embedding #{i} / {len(ds_idxs)}")

        embeds = np.vstack(embeds)
        
        if do_norm:
            embeds /= np.sqrt(np.square(embeds).sum(axis=1)).reshape(-1,1)

        return embeds

if __name__ == '__main__':
    import time
    master_ec_path = '../data/master_ec_idxs.csv'
    master_ec_df = pd.read_csv(master_ec_path, delimiter='\t')
    master_ec_idxs = {k: i for i, k in enumerate(master_ec_df.loc[:, 'EC number'])}

    X_name, Y_name = 'price', 'price'
    embed_type = 'clean'
    kfold = 5
    gs_dict = {'ks':[1,]}
    cf_model = cf(X_name, Y_name, embed_type, master_ec_idxs) # Init
    cf_model.fit() # Fit
    res = cf_model.kfold_knn_opt(kfold, gs_dict) # k-fold knn opt
    print(res)

    # X_name, Y_name = 'swissprot', 'price'
    # embed_type = 'esm'
    # k = 3
    # decision_threshold = 0.41
    # cf_model = cf(X_name, Y_name, embed_type, master_ec_idxs) # Init
    # cf_model.fit() # Fit
    # Y_hat = cf_model.predict(k)
    # Y_pred = Y_hat > decision_threshold
    # metric = cf_model.evaluate(Y_pred, f1_score)
    # print(metric)

    # tic = time.perf_counter()
    # pass
    # toc = time.perf_counter()
    # print(f"{toc - tic:.2f} s")

