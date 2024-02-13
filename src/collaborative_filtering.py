'''
to do

make x and y idxs into elts of a dict indexed by x and y names
generalize batch_construct_sim to take left name, right name
write def hypertune() which calls batch_construct sim with X twice
...follow through rest of predict to make sure works

have to address this upper bound on k due to batch size...
'''

import os
import pandas as pd
import subprocess
import numpy as np
from src.utils import load_embed
import scipy as sp

class cf:
    
    def __init__(self, X_name:str, Y_name:str, sample_embeds,
                master_feature_idxs:dict, feature_feature_sim_mats=[],
                batch_size=1500
                ):
        self.X_name = X_name
        self.Y_name = Y_name
        self.ffsm = feature_feature_sim_mats
        self.sample_embeds = sample_embeds
        self.feature_idxs = master_feature_idxs
        self.X_idxs = None
        self.Y_idxs = None
        self.batch_size = batch_size

        # Get id to index for samples and features
        self.X_idxs, self.nk = self._read_toc(self.X_name, return_idxs=True) # nk number known samples
        self.Y_idxs, self.nu = self._read_toc(self.Y_name, return_idxs=True) # nu number of unknown samples

        # TODO: run read_toc for all ffsm

        self.n_batches = self.nk // self.batch_size + 1 # Batch along nk >> nu
        self.nf = len(self.feature_idxs)


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
        to_construct = []
        for se in self.sample_embeds:
            path_pref = self.get_sim_mat_path_pref(se, self.X_name, self.Y_name)
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
            X_embeds = self.load_dense_embeds(self.X_name, self.X_idxs, se)
            Y_embeds = self.load_dense_embeds(self.Y_name, self.Y_idxs, se)

            # Matmul and save sim_batch
            print("Saving similarity matrices")
            for i in range(self.n_batches):
                path_pref = self.get_sim_mat_path_pref(se, self.X_name, self.Y_name)
                sim_batch = X_embeds[i * self.batch_size : (i + 1) * self.batch_size] @ Y_embeds.T
                np.save(path_pref + f"_batch_{i}.npy", sim_batch)
            
    def get_sparse_adj_mat(self, ds_name, row_idxs, col_idxs):
        path = self.get_adj_mat_path(ds_name)
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
                    j = self.feature_idxs[ec]
                    row.append(i)
                    col.append(j)
                    data.append(1)
                
                print(f"{i}/{shape[0]}", end='\r')

            adj = sp.sparse.csr_array((data, (row, col)), shape=shape)
            print("\nSaving sparse adjacency matrix")
            sp.sparse.save_npz(path, adj)

        return adj
    
    def _batch_knn_and_norm(self, path_pref, k):
        threshes = np.zeros(shape=(k, self.nu))
        sums = np.zeros(shape=(1, self.nu))
        for i in range(self.n_batches):
            path = path_pref + f"_batch_{i}.npy"
            sim_mat_i = np.load(path)

            threshes_i = np.sort(sim_mat_i, axis=0)[-k:, :]
            threshes = np.sort(np.vstack((threshes, threshes_i)), axis=0)[-k:, :]

            sums += sim_mat_i.sum(axis=0).reshape(1, -1)

        threshes = threshes[-k, :].reshape(1,-1)

        return threshes, sums


    def predict(self, k, embed_type):
        # TODO: Bring in feature side / ensembling
        # for se in self.sample_embeds:
        self.batch_construct_dense_sim_mats()
        X = self.get_sparse_adj_mat(self.X_name, self.X_idxs, self.feature_idxs)
        Y_hat = sp.sparse.csr_matrix((self.nf, self.nu)) # Init empty sparse arr
        path_pref = self.get_sim_mat_path_pref(embed_type, self.X_name, self.Y_name)
        
        k_thresholds, sums = self._batch_knn_and_norm(path_pref, k)
        
        for i in range(self.n_batches):
            path = path_pref + f"_batch_{i}.npy"
            sim_mat_i = np.load(path)

            sim_mat_i[sim_mat_i < k_thresholds] = 0 # k-threshold
            sim_mat_i /= sums # Normalize
            
            Y_hat += sp.sparse.csr_array(X.T[:, i * self.batch_size : (i + 1) * self.batch_size] @ sim_mat_i)

        return Y_hat.T.toarray()

    def evaluate(self, k, embed_type, test):
        Y_hat = self.predict(k, embed_type)
        Y_true = self.get_sparse_adj_mat(self.Y_name, self.Y_idxs, self.feature_idxs)
        Y_true = Y_true.toarray()

        # ROC doesn't like when no groud truth sample in a class
        in_sample_classes = np.where(Y_true.sum(axis=0) > 0)[0]
        Y_true = Y_true[:, in_sample_classes]
        Y_hat = Y_hat[:, in_sample_classes]

        metric = test(Y_true.ravel(), Y_hat.ravel())
        return metric

    
    def get_adj_mat_path(self, ds_name):
        return f"../data/{ds_name}/cf_adj_mat.npz"

    def get_sim_mat_path_pref(self, type, left, right):
        return f"../data/sim_mats/{type}_{left}_{right}"
    
    def load_dense_embeds(self, ds_name, ds_idxs, embed_type, do_norm=True):
        magic_key = 33
        data_path = f"../data/{ds_name}/"
        embeds = []
        for elt in ds_idxs:
            embeds.append(load_embed(data_path + f"{embed_type}/{elt}.pt", embed_key=magic_key)[1])

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
    k = 1000

    master_ec_df = pd.read_csv(master_ec_path, delimiter='\t')
    master_ec_idxs = {k: i for i, k in enumerate(master_ec_df.loc[:, 'EC number'])}

    cf = cf(X_name, Y_name, sample_embeds, master_ec_idxs)
    # y_hat = cf.predict(k, 'esm')
    metric = cf.evaluate(k, sample_embeds[0], roc_auc_score)
    print(metric)

    # tic = time.perf_counter()
    # pass
    # toc = time.perf_counter()
    # print(f"{toc - tic:.2f} s")

