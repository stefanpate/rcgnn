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

class cf:
    
    def __init__(self, X_name:str, Y_name:str, sample_embeds,
                master_feature_idxs:dict, feature_feature_sim_mats=[],
                batch_size=15000, seed=1234
                ):
        self.X_name = X_name
        self.Y_name = Y_name
        self.ffsm = feature_feature_sim_mats
        self.sample_embeds = sample_embeds
        self.idxs = {}
        self.idxs['feature'] = master_feature_idxs
        self.X_idxs = None
        self.Y_idxs = None
        self.batch_size = batch_size
        self.shapes = {}
        self.seed = seed

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

    
    def batch_construct_dense_sim_mats(self, left_name, right_name): 
        # Check for existing sim_mats
        print("Checking for similarity matrices")
        to_construct = []
        for se in self.sample_embeds:
            path_pref = self.get_sim_mat_path_pref(se, left_name, right_name)
            sim_mat_dir = '/'.join(path_pref.split('/')[:-1])
            fn_pref = path_pref.split('/')[-1]
            matches = [elt for elt in os.listdir(sim_mat_dir) if fn_pref in elt]

            if len(matches) != self.n_batches:
                for elt in matches:
                    subprocess.run(["rm", f"{sim_mat_dir}/{elt}"])

                to_construct.append(se)

        # Construct
        print("Constructing similarity matrices")
        for se in to_construct:
            # Load embeds
            left_embeds = self.load_dense_embeds(left_name, se)

            # Avoid loading same thing twice
            if left_name == right_name:
                right_embeds = left_embeds
            else:
                right_embeds = self.load_dense_embeds(right_name, se)

            # Matmul and save sim_batch
            print("Saving similarity matrices")
            for i in range(self.n_batches):
                path_pref = self.get_sim_mat_path_pref(se, left_name, right_name)
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
    
    def _kfold_splits(self, kfold):
        rng = np.random.default_rng(seed=self.seed)
        cv_idxs = np.arange(self.shapes[self.X_name][0])
        fold_size = int(self.shapes[self.X_name][0] / kfold)
        rng.shuffle(cv_idxs)
        splits = [cv_idxs[i * fold_size : (i + 1) * fold_size] for i in range(kfold)]
        return splits
    
    def _batch_knn(self, path_pref, ks, kfold=1):
        print("kNN thresholding & normalizing")
        max_k = max(ks) # k-thresholds for lesser ks come along with max_k
        for i in range(self.n_batches):
            # print(f"Batch {i}")

            # Load batch sim mat
            path = path_pref + f"_batch_{i}.npy"
            batch_sim_mat = np.load(path)

            if i == 0:
                d = batch_sim_mat.shape[1] # Infer last dimension from 1st batch
                max_threshes = np.zeros(shape=(kfold, max_k, d)) # Create 3d array to store n_splits (max_k x d) "top-k" matrices             

            for j, split in enumerate(self._kfold_splits(kfold)):

                    # Zero out rows & cols at in case of HPO (kfold > 1)
                    batch_sim_mat, row_vals, col_vals = self._mask_at_idxs(batch_sim_mat, split, batch_no=i)

                    # Update this split's max_threshes top-k matrix
                    batch_split_threshes = np.sort(batch_sim_mat, axis=0)[-max_k:, :]
                    max_threshes[j] = np.sort(np.vstack((max_threshes[j], batch_split_threshes)), axis=0)[-max_k:, :]

                    # Put back masked values
                    batch_sim_mat = self._unmask_at_idxs(batch_sim_mat, split, row_vals, col_vals, batch_no=i)

        # Create 3d array object to return (ks (as in kNN ks), kfold (# cv splits), d)
        threshes = np.zeros(shape=(len(ks), kfold, d)) # kNN threshold vals
        for i, k in enumerate(ks):
            for j in range(kfold):
                threshes[i, j, :] = max_threshes[j, -k, :]

        return threshes
    
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
    
    def _kfold_predict(self, X, path_pref, ks, embed_type, kfold):
        # TODO: Bring in feature side / ensembling
        # for se in self.sample_embeds:
        X_hats = defaultdict(lambda : sp.sparse.csr_array(X.shape))
        threshes = self._batch_knn(path_pref, ks, kfold)
        ks = sorted(ks, reverse=True) # Go in descending order so don't need to hold onto threshed out values
        
        print("Predicting")
        for i in range(self.n_batches):
            # Load batch sim mat
            path = path_pref + f"_batch_{i}.npy"
            batch_sim_mat = np.load(path)

            if i == 0:
                d = batch_sim_mat.shape[1] # Infer last dimension from 1st batch
                sums = defaultdict(lambda : np.zeros(shape=(1, d))) # Store sums for each kNN-split condition

            for n, k in enumerate(ks):             
                for j, split in enumerate(self._kfold_splits(kfold)):

                        batch_sim_mat, row_vals, col_vals = self._mask_at_idxs(batch_sim_mat, split, batch_no=i) # Zero out rows & cols at in case of HPO (kfold > 1)
                        knn_split_threshes = threshes[n, j, :].reshape(1, -1) # To threshold cols of batch sim mat
                        batch_sim_mat[batch_sim_mat < knn_split_threshes] # k-threshold

                        # Update X_hat & sums
                        X_hats[(k, j)] += sp.sparse.csr_array(X.T[:, i * self.batch_size : (i + 1) * self.batch_size] @ batch_sim_mat).T
                        sums[(k, j)] += batch_sim_mat.sum(axis=0).reshape(1, -1)

                        # Put back masked values
                        batch_sim_mat = self._unmask_at_idxs(batch_sim_mat, split, row_vals, col_vals, batch_no=i)
            
        # Normalize (taking into acct k-thresholding)
        for key in X_hats.keys():
            X_hats[key] = X_hats[key] / sums[key].T

        return X_hats

    def predict(self, left_name, right_name, k, embed_type, cv_idxs=None):
        # TODO: Bring in feature side / ensembling
        # for se in self.sample_embeds:
        self.batch_construct_dense_sim_mats(left_name, right_name)
        left = self.get_sparse_adj_mat(left_name)
        right_hat = sp.sparse.csr_array(self.shapes[right_name]) # Init empty sparse arr
        path_pref = self.get_sim_mat_path_pref(embed_type, left_name, right_name)
        
        k_thresholds, sums = self._batch_knn_and_norm(path_pref, k)
        
        print("Predicting")
        for i in range(self.n_batches):
            # print(f"Batch {i}")
            path = path_pref + f"_batch_{i}.npy"
            sim_mat_i = np.load(path)

            # Zero out rows & cols at cv_idxs (e.g., HPO)
            if cv_idxs is not None:
                sim_mat_i = self._mask_at_idxs(sim_mat_i, cv_idxs, batch_no=i)

            sim_mat_i[sim_mat_i < k_thresholds] = 0 # k-threshold
            sim_mat_i /= sums # Normalize
            
            right_hat += sp.sparse.csr_array(left.T[:, i * self.batch_size : (i + 1) * self.batch_size] @ sim_mat_i).T

        return right_hat.toarray()

    def evaluate(self, X_true, X_hat, test):

        # ROC doesn't like when no groud truth sample in a class
        in_sample_classes = np.where(X_true.sum(axis=0) > 0)[0]
        X_true = X_true[:, in_sample_classes]
        X_hat = X_hat[:, in_sample_classes]

        metric = test(X_true.ravel(), X_hat.ravel())
        return metric
    
    def kfold_knn_opt(self, X_name, kfold, embed_type, grid_search:dict):
        ks = grid_search['ks']
        X = self.get_sparse_adj_mat(X_name)
        path_pref = self.get_sim_mat_path_pref(embed_type, X_name, X_name)
        X_hats = self._kfold_predict(X, path_pref, ks, embed_type, kfold)
        splits = self._kfold_splits(kfold)

        res = defaultdict(list)
        for i, k in enumerate(ks):
            f1s, decision_thresholds = [], []
            for j, split in enumerate(splits):

                # Compute best F1 w/ decision threshold
                X_true = X[split, :].toarray()
                X_hat = X_hats[k, j].toarray()[split, :]
                metric = self.evaluate(X_true, X_hat, precision_recall_curve)
                precision, recall, dts = metric
                f1s.append(np.max(np.sqrt(recall * precision)))
                decision_thresholds.append(dts[np.argmax(np.sqrt(recall * precision))])

            # Save split-averaged measures
            f1s = np.array(f1s)
            decision_thresholds = np.array(decision_thresholds)
            res["k1"].append(k)
            res["F1"].append(f1s.mean())
            res["F1_err"].append(f1s.std() / np.sqrt(kfold))
            res["thresholds"].append(decision_thresholds.mean())
            res["thresholds_err"].append(decision_thresholds.std() / np.sqrt(kfold))

            # Save
            save_json(res, f"../artifacts/cf/{kfold}_fold_hpo_{X_name}_{embed_type}_new.json")
            print(f"Last k saved: {k}, # {i + 1} / {len(ks)}")

        return res


        # gs_keys = list(grid_search.keys())
        # gs_values = list(grid_search.values())
        # hyperparams = list(product(*gs_values))
        # res = defaultdict(list)
        # # TODO: use gs vals and keys to do multiple hypparams
        # for j, elt in enumerate(hyperparams):

        #     # Enter next tuple of hp values
        #     for i in range(len(elt)):
        #         res[gs_keys[i]].append(elt[i])
            
        #     inside_F1s = []
        #     inside_thresholds = []
        #     for i in range(kfold):
        #         metric = self.evaluate(self.X_name, self.X_name, elt[0],
        #                                 embed_type, precision_recall_curve,
        #                                 cv_idxs[i * fold_size : (i + 1) * fold_size]
        #                                 )
        #         precision, recall, thresholds_pr = metric
        #         best_F1 = np.max(np.sqrt(recall * precision))
        #         best_threshold = thresholds_pr[np.argmax(np.sqrt(recall * precision))]
        #         inside_F1s.append(best_F1)
        #         inside_thresholds.append(best_threshold)

        #     # Enter performance metrics
        #     inside_thresholds, inside_F1s = np.array(inside_thresholds), np.array(inside_F1s)
        #     res["F1"].append(inside_F1s.mean())
        #     res["F1_err"].append(inside_F1s.std() / np.sqrt(kfold))
        #     res["thresholds"].append(inside_thresholds.mean())
        #     res["thresholds_err"].append(inside_thresholds.std() / np.sqrt(kfold))

        #     # Save
        #     save_json(res, f"../artifacts/cf/{kfold}_fold_hpo_{X_name}_{sample_embeds[0]}.json")
        #     print(f"Last hyperparameter set save: {gs_keys} = {elt}, # {j} / {len(hyperparams)}")

        # return res

    def get_adj_mat_path(self, ds_name):
        return f"../data/{ds_name}/cf_adj_mat.npz"

    def get_sim_mat_path_pref(self, type, left, right):
        # return f"../data/sim_mats/{type}_{left}_{right}"
        # return f"/media/stef/EXTHD/hiec_scratch/{type}_{left}_{right}"
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
    from sklearn.metrics import roc_auc_score, accuracy_score
    X_name, Y_name = 'swissprot', 'price'
    sample_embeds = ['clean']
    master_ec_path = '../data/master_ec_idxs.csv'

    master_ec_df = pd.read_csv(master_ec_path, delimiter='\t')
    master_ec_idxs = {k: i for i, k in enumerate(master_ec_df.loc[:, 'EC number'])}

    cf_model = cf(X_name, Y_name, sample_embeds, master_ec_idxs)
    kfold = 5
    cf_model.batch_construct_dense_sim_mats(X_name, X_name)
    gs_dict = {'ks':[3,]}
    res = cf_model.kfold_knn_opt(X_name, kfold, sample_embeds[0], gs_dict)

    # y_hat = cf.predict(X_name, Y_name, k, 'esm')
    # metric = cf.evaluate(X_name, Y_name, k, sample_embeds[0], roc_auc_score)
    # metric = cf.evaluate(X_name, X_name, k, sample_embeds[0], roc_auc_score, np.array([0,1,2,3]))
    # print(metric)

    # metric = cf_model.evaluate('new', 'new', k, sample_embeds[0], precision_recall_curve, np.random.random_integers(0, 391, size=(78,)))
    # precision, recall, thresholds_pr = metric
    # best_f1 = np.max(np.sqrt(recall * precision))
    # best_threshold = thresholds_pr[np.argmax(np.sqrt(recall * precision))]
    # print(f"Max F1:{best_f1}")
    # print(f"Best decision threshold: {best_threshold}")

    # tic = time.perf_counter()
    # pass
    # toc = time.perf_counter()
    # print(f"{toc - tic:.2f} s")

