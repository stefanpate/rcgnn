'''
Grid search for matrix factorization using skorch
'''
import numpy as np
import torch
import multiprocessing as mp

seed = 1234
rng = np.random.default_rng(seed=seed) # Seed random number generator
    
def negative_sample_bipartite(n_samples, n_rows, n_cols, obs_pairs):
    # Sample subset of unobserved pairs
    unobs_pairs = []
    while len(unobs_pairs) < n_samples:
        i = rng.integers(0, n_rows)
        j = rng.integers(0, n_cols)

        if (i, j) not in obs_pairs:
            unobs_pairs.append((i, j))

    return unobs_pairs

def fit_eval(q_in, q_out, lock):
    while True:
        elt = q_in.get()
        
        if elt is None:
            break
        
        model, X_train, y_train, hp, split_no, hp_no = elt # hp is a dict, split_no = which cv split

        with lock:
            print(f"Fitting: Split #{split_no} of {hp}")
        
        fit_tic = time.perf_counter()
        model.fit(X_train, y_train)
        fit_toc = time.perf_counter()
        fit_time = fit_toc - fit_tic

        with lock:
            print(f"Fit Split #{split_no} of {hp} in {fit_time:.2f} seconds")
        
        # Get losses
        train_loss, val_loss = model.history[-1, ("train_loss", "valid_loss")]
        q_out.put((train_loss, val_loss, hp, split_no, hp_no, fit_time))


if __name__ == '__main__':
    from src.mf import MatrixFactorization
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
    qsize = 1
    n_cores = 3
    neg_multiple = 1 # How many negative samples per
    n_splits = 3
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    custom_nll = make_scorer(log_loss, labels=[0., 1.], greater_is_better=False, needs_proba=True)
    refit = True # Refit best model from grid search on all data

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

    # Split data for cv
    neg_pairs = []
    train_idxs = []
    test_idxs = []
    n_obs = len(obs_pairs)
    for train, test in kfold.split(obs_pairs):
        test_idxs.append(test)
        train_obs = [obs_pairs[elt] for elt in train]
        n_neg = train.shape[0] * neg_multiple
        nps = negative_sample_bipartite(n_neg, n_users, n_items, train_obs) # Sample negatives
        ntis = np.array([i + n_obs for i in range(len(nps))])
        neg_pairs += nps
        train_idxs.append(np.hstack((train, ntis)))

    # Data
    X = np.vstack((obs_pairs, neg_pairs)).astype(np.int64)
    y = np.vstack((np.ones(shape=(len(obs_pairs), 1)), np.zeros(shape=(len(neg_pairs), 1)))).astype(np.float32)

    # cv splits
    cv_idxs = list(zip(train_idxs, test_idxs))

    # Hyperparams for grid search
    # hps = {
    #     'lr':[5e-3,],
    #     'max_epochs':[5000, 7500],
    #     'batch_size':[1, 10, 25],
    #     'optimizer__weight_decay':[5e-3],
    #     'module__n_factors':[10, 100],
    #     'module__scl_embeds':[False]
    # }

    hps = {
        'lr':[5e-3,],
        'max_epochs':[5000,],
        'batch_size':[10,],
        'optimizer__weight_decay':[5e-3],
        'module__n_factors':[10,],
        'module__scl_embeds':[False]
    }

    # Cartesian product of hyperparams
    hp_keys = list(hps.keys())
    hp_cart_prod = [{hp_keys[i] : elt[i] for i in range(len(elt))} for elt in product(*hps.values())]

    # Initialize queues and pool
    to_fit = mp.Queue(maxsize=qsize)
    fitted = mp.Queue()
    the_lock = mp.Lock()
    pool = mp.Pool(n_cores,
                   initializer=fit_eval,
                   initargs=(to_fit, fitted, the_lock)
                   )
    
    # Parallel grid search
    gs_tic = time.perf_counter()
    for hp_no, hp in enumerate(hp_cart_prod):
        for split_no, split in enumerate(cv_idxs):

            # Prep data
            train, val = split
            X_train, y_train = X[train], y[train]
            X_val, y_val = X[val], y[val]
            val_ds = Dataset(X_val, y_val) # Predefined validation split

            # Construct model
            model = NeuralNetClassifier(
                module=MatrixFactorization,
                criterion=torch.nn.BCELoss(),
                optimizer=torch.optim.SGD,
                device=device,
                module__n_users=n_users,
                module__n_items=n_items,
                train_split=predefined_split(val_ds),
                **hp
            )

            # Populate in-queue
            to_fit.put((model, X_train, y_train, hp, split_no, hp_no))

    # Signal to workers: 'all done'
    for _ in range(n_cores):
        to_fit.put(None)

    pool.close()
    pool.join()

    # Assemble gs results
    gs_res = {
        'train_loss':[], 'val_loss':[], 'fit_time':[],
        'hp_no':[], 'split_no':[],
        **{k:[] for k in hps.keys()}
    }
    
    while not fitted.empty():
        train_loss, val_loss, hp, split_no, hp_no, fit_time = fitted.get()
        for k,v  in hp.items():
            gs_res[k].append(v)

        gs_res['train_loss'].append(train_loss)
        gs_res['val_loss'].append(val_loss)
        gs_res['fit_time'].append(fit_time)
        gs_res['hp_no'].append(hp_no)
        gs_res['split_no'].append(split_no)

    df = pd.DataFrame(gs_res)

    # Average and std of losses and fit times
    agg_dict = {
        'train_loss':['mean', 'std'], 'val_loss':['mean', 'std'],
        'fit_time':'mean', **{k:'first' for k in hps.keys()}
    } 
    gs_res = df.groupby("hp_no").agg(agg_dict)
    gs_res.columns = ['mean_train_loss', "std_train_loss",
                  "mean_val_loss", "std_test_loss",
                  "mean_fit_time"] + [k for k in hps.keys()]
    gs_res.reset_index(inplace=True)
    gs_res.sort_values(by=['mean_val_loss'], inplace=True, ascending=True)

    # Save grid search results
    gs_res.to_csv(f"{eval_models_dir}/{timestamp}_grid_search_{dataset_name}_neg_multiple_{neg_multiple}.csv", sep='\t')
    gs_toc = time.perf_counter()
    print(f"Grid search on {n_cores} cores took {gs_toc - gs_tic:.2f} seconds")  
    
    # Re-fit on entire dataset
    if refit:
        best_params = gs_res.loc[0,:].to_dict()
        best_params = {k:best_params[k] for k in hps.keys()} # Subselect

        # Split observed positives
        X_train, X_test, y_train, y_test = train_test_split(obs_pairs, np.ones(shape=(len(obs_pairs), 1)),
                                                            train_size=0.8,
                                                            shuffle=True,
                                                            random_state=seed)
        
        # Sample negatives
        n_neg = len(X_train) * neg_multiple
        nps = negative_sample_bipartite(n_neg, n_users, n_items, X_train) # Sample negatives

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

        best_model = NeuralNetClassifier(
            module=MatrixFactorization,
            criterion=torch.nn.BCELoss(),
            optimizer=torch.optim.SGD,
            device=device,
            module__n_users=n_users,
            module__n_items=n_items,
            train_split=predefined_split(val_ds),
            **best_params
        )

        best_model.fit(X_train, y_train)

        # Save best_model
        with open(f"{trained_models_dir}/{timestamp}_best_model_{dataset_name}_neg_multiple_{neg_multiple}_seed_{seed}.pkl", 'wb') as f:
            pickle.dump(best_model, f)

    print('done')