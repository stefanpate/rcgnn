'''
Grid search for matrix factorization using skorch
'''
import numpy as np
import torch
import multiprocessing as mp

seed = 1234
rng = np.random.default_rng(seed=seed) # Seed random number generator


if __name__ == '__main__':
    from src.mf import MatrixFactorization, negative_sample_bipartite
    from skorch import NeuralNetClassifier
    from skorch.dataset import Dataset
    from skorch.helper import predefined_split
    from sklearn.model_selection import KFold, GridSearchCV, train_test_split
    from sklearn.metrics import log_loss, make_scorer
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from src.utils import ensure_dirs, construct_sparse_adj_mat
    import pickle
    import time
    from itertools import product

    seed = 1234
    rng = np.random.default_rng(seed=seed) # Seed random number generator
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%y%m%d_%H_%M_%S")

    # Data parameters
    n_users = 1_00
    n_items = 1_00
    k = 2 # K nearest neighbors to "link" in adj matrix
    density = 0.01 # For random adj mat
    dataset_name = 'sp_ops' # random | knn | one of the datasets

    # Training parameters
    neg_multiple = 1 # How many negative samples per
    custom_nll = make_scorer(log_loss, labels=[0., 1.], greater_is_better=False, needs_proba=True)
    warm_start = True # Continue training model where you left off
    ws_fn = "240429_12_21_45_fit_model_sp_ops_neg_multiple_1_seed_1234.pkl" # Filenema of desired warm start model
    ws_n_epochs = 2500

    # Saving parameters
    trained_models_dir = "/projects/p30041/spn1560/hiec/artifacts/trained_models/mf"
    eval_models_dir = "../artifacts/model_evals/mf"
    ensure_dirs(trained_models_dir)
    ensure_dirs(eval_models_dir)

    # Generate adjacency matrix
    if dataset_name == 'knn':
        obs_pairs = []
        for i in range(n_users):
            for j in range(max(0, i - k), min(i + k, n_items)):
                obs_pairs.append((i, j))

        ratings = np.zeros(shape=(n_users, n_items))
        nz_rows, nz_cols = [np.array(elt) for elt in list(zip(*obs_pairs))]
        ratings[nz_rows, nz_cols] = 1

    elif dataset_name == 'random':
        n_obs = int(n_users * n_items * density)
        obs_rows = rng.integers(0, n_users, size=(n_obs,))
        obs_cols = rng.integers(0, n_items, size=(n_obs,))
        obs_pairs = list(zip(obs_rows, obs_cols))
        ratings = np.zeros(shape=(n_users, n_items))
        ratings[obs_rows, obs_cols] = 1

    else:
        ratings, idx_sample, idx_feature = construct_sparse_adj_mat(dataset_name)
        obs_pairs = list(zip(*ratings.nonzero()))
        n_users, n_items = ratings.shape

    hps = {
        'lr':5e-3,
        'max_epochs':2000,
        'batch_size':10,
        'optimizer__weight_decay':1e-4,
        'module__n_factors':20,
        'module__scl_embeds':False
    }

    # Split observed positives
    X_train, X_test, y_train, y_test = train_test_split(obs_pairs, np.ones(shape=(len(obs_pairs), 1)),
                                                        train_size=0.8,
                                                        shuffle=True,
                                                        random_state=seed)
    
    # Sample negatives
    n_neg = len(X_train) * neg_multiple
    nps = negative_sample_bipartite(n_neg, n_users, n_items, X_train, seed=seed) # Sample negatives

    # Enforce right type
    X_train = np.array(X_train + nps).astype(np.int64)
    y_train = np.concatenate((y_train, np.zeros(shape=(len(nps), 1)))).astype(np.float32)
    X_test = np.array(X_test).astype(np.int64)
    y_test = y_test.astype(np.float32)

    # Shuffle traindata with negatives
    p = rng.permutation(X_train.shape[0])
    X_train = X_train[p]
    y_train = y_train[p]
    val_ds = Dataset(X_test, y_test) # Predefined validation split

    # Load  or construct model and fit
    if warm_start:
        with open(f"{trained_models_dir}/{ws_fn}", 'rb') as f:
            model = pickle.load(f)

        model.warm_start = warm_start
        model.max_epochs = ws_n_epochs

    else:
        model = NeuralNetClassifier(
            module=MatrixFactorization,
            criterion=torch.nn.BCELoss(),
            optimizer=torch.optim.SGD,
            device=device,
            module__n_users=n_users,
            module__n_items=n_items,
            train_split=predefined_split(val_ds),
            **hps
        )

    tic = time.perf_counter()
    model.fit(X_train, y_train)
    toc = time.perf_counter()

    print(f"{hps['max_epochs']} epochs took {toc - tic:.2f} seconds")

    # Save model
    with open(f"{trained_models_dir}/{timestamp}_fit_model_{dataset_name}_neg_multiple_{neg_multiple}_seed_{seed}.pkl", 'wb') as f:
        pickle.dump(model, f)

    print('done')