from src.utils import load_embed
import numpy as np
import os

def load_design_matrix(ds_name, embed_type, sample_idx, do_norm=False):
        '''
        x

        Args
            - ds_name: Str name of dataset
            - embed_type: Str
            - sample_idx: {sample_label : row_idx}

        Returns
            - X: Design matrixs (samples x embedding dim)
        '''
        # Load from scratch if pre-saved
        path = f"/scratch/spn1560/{ds_name}_{embed_type}_X_unnormed.npy"
        if os.path.exists(path):
            X = np.load(path)
        else:

            print(f"Loading {embed_type} embeddings for {ds_name} dataset")
            magic_key = 33
            data_path = f"../data/{ds_name}/"
            X = []
            for i, elt in enumerate(sample_idx):
                X.append(load_embed(data_path + f"{embed_type}/{elt}.pt", embed_key=magic_key)[1])

                if i % 5000 == 0:
                    print(f"Embedding #{i} / {len(sample_idx)}")

            X = np.vstack(X)
            
            if do_norm:
                X /= np.sqrt(np.square(X).sum(axis=1)).reshape(-1,1)

            # Save to scratch
            np.save(path, X)

        return X


if __name__ == '__main__':
    from src.utils import construct_sparse_adj_mat
    train_data_name = 'swissprot'
    embed_type = 'esm'
    y, idx_sample, idx_feature = construct_sparse_adj_mat(train_data_name)
    sample_idx = {v:k for k,v in idx_sample.items()}
    X = load_design_matrix(train_data_name, embed_type, sample_idx)