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
                batch_size=15000
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
    
    def _batch_knn_and_norm(self, path_pref, k, cv_idxs=None):
        print("kNN thresholding & normalizing")
        for i in range(self.n_batches):
            # print(f"Batch {i}")

            path = path_pref + f"_batch_{i}.npy"
            sim_mat_i = np.load(path)

            # Zero out rows & cols at cv_idxs (e.g., HPO)
            if cv_idxs is not None:
                sim_mat_i = self._zero_at_idxs(sim_mat_i, cv_idxs, batch_no=i)

            # Infer shape from first batch
            if i == 0:
                threshes = np.zeros(shape=(k, sim_mat_i.shape[1]))
                sums = np.zeros(shape=(1, sim_mat_i.shape[1]))
            
            threshes_i = np.sort(sim_mat_i, axis=0)[-k:, :]
            threshes = np.sort(np.vstack((threshes, threshes_i)), axis=0)[-k:, :]

            sums += sim_mat_i.sum(axis=0).reshape(1, -1)

        threshes = threshes[-k, :].reshape(1,-1)

        return threshes, sums
    
    def _zero_at_idxs(self, mat, idxs, batch_no=None):
        if batch_no is not None:
            lb, ub = batch_no * self.batch_size, (batch_no + 1) * self.batch_size
            idxs = idxs[(idxs > lb) & (idxs < ub)]
            idxs = idxs % self.batch_size

        mat[idxs, :] = 0
        mat[:, idxs] = 0

        return mat

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
                sim_mat_i = self._zero_at_idxs(sim_mat_i, cv_idxs, batch_no=i)

            sim_mat_i[sim_mat_i < k_thresholds] = 0 # k-threshold
            sim_mat_i /= sums # Normalize
            
            right_hat += sp.sparse.csr_array(left.T[:, i * self.batch_size : (i + 1) * self.batch_size] @ sim_mat_i).T

        return right_hat.toarray()

    def evaluate(self, left_name, right_name, k, embed_type, test, cv_idxs=None):
        Y_hat = self.predict(left_name, right_name, k, embed_type, cv_idxs)
        Y_true = self.get_sparse_adj_mat(right_name)
        Y_true = Y_true.toarray()

        # Pick out test set if HPO
        if cv_idxs is not None:
            Y_true = Y_true[cv_idxs, :]
            Y_hat = Y_hat[cv_idxs, :]

        # ROC doesn't like when no groud truth sample in a class
        in_sample_classes = np.where(Y_true.sum(axis=0) > 0)[0]
        Y_true = Y_true[:, in_sample_classes]
        Y_hat = Y_hat[:, in_sample_classes]

        metric = test(Y_true.ravel(), Y_hat.ravel())
        return metric
    
    def kfold_knn_opt(self, kfold, embed_type, grid_search:dict, seed=1234):
        rng = np.random.default_rng(seed=seed)
        cv_idxs = np.arange(self.shapes[self.X_name][0])
        rng.shuffle(cv_idxs)
        gs_keys = list(grid_search.keys())
        gs_values = list(grid_search.values())
        hyperparams = list(product(*gs_values))
        fold_size = int(self.shapes[self.X_name][0] / kfold)

        res = defaultdict(list)
        # TODO: use gs vals and keys to do multiple hypparams
        for j, elt in enumerate(hyperparams):

            # Enter next tuple of hp values
            for i in range(len(elt)):
                res[gs_keys[i]].append(elt[i])
            
            inside_F1s = []
            inside_thresholds = []
            for i in range(kfold):
                metric = self.evaluate(self.X_name, self.X_name, elt[0],
                                        embed_type, precision_recall_curve,
                                        cv_idxs[i * fold_size : (i + 1) * fold_size]
                                        )
                precision, recall, thresholds_pr = metric
                best_F1 = np.max(np.sqrt(recall * precision))
                best_threshold = thresholds_pr[np.argmax(np.sqrt(recall * precision))]
                inside_F1s.append(best_F1)
                inside_thresholds.append(best_threshold)

            # Enter performance metrics
            inside_thresholds, inside_F1s = np.array(inside_thresholds), np.array(inside_F1s)
            res["F1"].append(inside_F1s.mean())
            res["F1_err"].append(inside_F1s.std() / np.sqrt(kfold))
            res["thresholds"].append(inside_thresholds.mean())
            res["thresholds_err"].append(inside_thresholds.std() / np.sqrt(kfold))

            # Save
            save_json(res, f"../artifacts/cf/{kfold}_fold_hpo_{X_name}_{embed_type}.json")
            print(f"Last hyperparameter set save: {gs_keys} = {elt}, # {j} / {len(hyperparams)}")

        return res


    
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
    sample_embeds = ['esm']
    master_ec_path = '../data/master_ec_idxs.csv'
    # k = 3

    master_ec_df = pd.read_csv(master_ec_path, delimiter='\t')
    master_ec_idxs = {k: i for i, k in enumerate(master_ec_df.loc[:, 'EC number'])}

    cf_model = cf(X_name, Y_name, sample_embeds, master_ec_idxs)

    # y_hat = cf.predict(X_name, Y_name, k, 'esm')
    # metric = cf.evaluate(X_name, Y_name, k, sample_embeds[0], roc_auc_score)
    # metric = cf.evaluate(X_name, X_name, k, sample_embeds[0], roc_auc_score, np.array([0,1,2,3]))
    # print(metric)

    kfold = 5
    gs_dict = {sample_embeds[0]:[1, 3, 5, 10, 50, 100]}
    res = cf_model.kfold_knn_opt(kfold, sample_embeds[0], gs_dict)

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

