'''
Fit matrix factorization implemented w/ skorch
'''

def generate_adjacency_matrix(mode, seed, n_users=100, n_items=100, k=2, density=0.01):
    rng = np.random.default_rng(seed=seed)
    if mode == 'knn':
        obs_pairs = []
        for i in range(n_users):
            for j in range(max(0, i - k), min(i + k, n_items)):
                obs_pairs.append((i, j))

        ratings = np.zeros(shape=(n_users, n_items))
        nz_rows, nz_cols = [np.array(elt) for elt in list(zip(*obs_pairs))]
        ratings[nz_rows, nz_cols] = 1

    elif mode == 'random':
        n_obs = int(n_users * n_items * density)
        obs_rows = rng.integers(0, n_users, size=(n_obs,))
        obs_cols = rng.integers(0, n_items, size=(n_obs,))
        obs_pairs = list(zip(obs_rows, obs_cols))
        ratings = np.zeros(shape=(n_users, n_items))
        ratings[obs_rows, obs_cols] = 1

    return obs_pairs, n_users, n_items
        
if __name__ == '__main__':
    from catalytic_function.mf import MatrixFactorization, PretrainedMatrixFactorization, load_pretrained_embeds
    from catalytic_function.utils import ensure_dirs, construct_sparse_adj_mat, load_data_split, load_hps_from_scratch, get_sample_feature_idxs, negative_sample_bipartite
    from skorch import NeuralNetClassifier
    from skorch.dataset import Dataset
    from skorch.helper import predefined_split
    from sklearn.model_selection import train_test_split
    import numpy as np
    import torch
    from datetime import datetime
    import time
    import pickle
    import json
    from argparse import ArgumentParser

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%y%m%d_%H_%M_%S")

    '''
    Set arguments
    '''

    # CLI parsing
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset-name", type=str)
    parser.add_argument("-e", "--seed", type=int)
    parser.add_argument("-n", "--n-splits", type=int)
    parser.add_argument("-s", "--split-idx", type=int)
    parser.add_argument("-p", "--hp-idx", type=int)
    parser.add_argument("-g", "--gs-name", type=str)
    parser.add_argument("-m", "--save-gs-models", action="store_true")

    args = parser.parse_args() # Parse cli args
    
    # CLI mode or not?
    if any([args.__dict__[k] is not None for k in args.__dict__.keys() if k != 'save_gs_models']):
        cli = True
    else:
        cli = False

    # Parse args in cli mode
    if cli:
        # Parse CLI arguments
        dataset_name = args.dataset_name
        seed = args.seed
        n_splits = args.n_splits
        split_idx = args.split_idx
        hp_idx = args.hp_idx
        gs_name = args.gs_name
        save_gs_models = args.save_gs_models

        hps = load_hps_from_scratch(gs_name, hp_idx) # Load hyperparams

    # Set args if not in cli mode
    else:
        seed = 1234
        dataset_name = 'sp_ops' # random | knn | one of the datasets

        # Set hyperparams
        hps = {
            'lr':5e-3,
            'max_epochs':1,
            'batch_size':10,
            'optimizer__weight_decay':1e-4,
            # 'module__n_factors':20,
            'module__scl_embeds':False,
            'neg_multiple':1,
            'user_embeds':"esm_rank_50"
        }

    # Set more args regardless of cli mode
    base_model = PretrainedMatrixFactorization
    warm_start = False # Continue training model where you left off
    ws_fn = "240429_12_21_45_fit_model_sp_ops_neg_multiple_1_seed_1234.pkl" # Filenema of desired warm start model
    ws_n_epochs = 2500

    # Saving parameters
    trained_models_dir = "/projects/p30041/spn1560/hiec/artifacts/trained_models/mf"
    gs_res_dir = "../artifacts/model_evals/mf/tmp"
    ensure_dirs(trained_models_dir)
    ensure_dirs(gs_res_dir)
    
    # Special hps that don't go (directly) to skorch model
    special_hps = ['neg_multiple', 'user_embeds', 'item_embeds']
    special_hps_for_gs = {k:hps.pop(k) if k in hps else None for k in special_hps}

    rng = np.random.default_rng(seed=seed) # Seed random number generator

    '''
    Arrange data
    '''
    print("Loading data")
    # Load positive data
    if cli:
        X_train, y_train, X_test, y_test = load_data_split(
            dataset_name,
            None,
            n_splits,
            seed,
            split_idx
        )

        idx_sample, idx_feature = get_sample_feature_idxs(dataset_name)
        n_users, n_items = len(idx_sample), len(idx_feature) # Size of adj mat depends on dataset

    else:
        adj, idx_sample, idx_feature = construct_sparse_adj_mat(dataset_name)
        obs_pairs = np.array(list(zip(*adj.nonzero())))
        # Split observed positives
        X_train, X_test, y_train, y_test = train_test_split(obs_pairs, np.ones(shape=(len(obs_pairs), 1)),
                                                            train_size=0.8,
                                                            shuffle=True,
                                                            random_state=seed)
        n_users, n_items = len(idx_sample), len(idx_feature) # Size of adj mat depends on dataset
    
    # TODO: remove this because im sampling negatives in batch_fit now
    # # Sample negatives
    # n_neg = len(X_train) * special_hps_for_gs['neg_multiple']
    # nps = negative_sample_bipartite(n_neg, n_users, n_items, X_train, seed=seed) # Sample negatives

    # # Add in negative samples and enforce the right data type
    # X_train = np.vstack((X_train, nps)).astype(np.int64)
    # y_train = np.concatenate((y_train, np.zeros(shape=(len(nps), 1)))).astype(np.float32)
    # X_test = np.array(X_test).astype(np.int64)
    # y_test = y_test.astype(np.float32)

    # Shuffle traindata with negatives
    p = rng.permutation(X_train.shape[0])
    X_train = X_train[p]
    y_train = y_train[p]
    val_ds = Dataset(X_test, y_test) # Predefined validation split

    '''
    Arrange special hps for model
    '''
    
    sample_idx = {v:k for k,v in idx_sample.items()}
    special_hps_for_model = load_pretrained_embeds(
        dataset_name,
        special_hps_for_gs,
    )

    if "module__n_users" in special_hps_for_model and "module__n_items" not in special_hps_for_model:
        special_hps_for_model["module__n_items"] = n_items
    elif "module__n_users" not in special_hps_for_model and "module__n_items" in special_hps_for_model:
        special_hps_for_model["module__n_users"] = n_users
    elif "module__n_users" not in special_hps_for_model and "module__n_items" not in special_hps_for_model:
        special_hps_for_model["module__n_users"] = n_users
        special_hps_for_model["module__n_items"] = n_items


    '''
    Load  or construct model and fit
    '''
    
    if warm_start: # Load previously trained model
        with open(f"{trained_models_dir}/{ws_fn}", 'rb') as f:
            model = pickle.load(f)

        model.warm_start = warm_start
        model.max_epochs = ws_n_epochs

    else: # Construct model
        model = NeuralNetClassifier(
            module=base_model,
            criterion=torch.nn.BCELoss(),
            optimizer=torch.optim.SGD,
            device=device,
            train_split=predefined_split(val_ds),
            **hps,
            **special_hps_for_model
        )

    print("Fitting model")
    tic = time.perf_counter()
    model.fit(X_train, y_train)
    toc = time.perf_counter()
    fit_time = toc - tic

    print(f"{hps['max_epochs']} epochs took {fit_time:.2f} seconds")

    '''
    Save
    '''
    print("Saving results")
    if cli:
        ds_curves = 250 # Downsample loss curves
        train_loss_curve, test_loss_curve = list(zip(*model.history[:, ('train_loss', 'valid_loss')]))
        train_loss, test_loss = train_loss_curve[-1], test_loss_curve[-1]
        train_loss_curve = [train_loss_curve[i] for i in range(0, len(train_loss_curve), ds_curves)]
        test_loss_curve = [test_loss_curve[i] for i in range(0, len(test_loss_curve), ds_curves)]
        gs_res = {
            'train_loss':train_loss, 'val_loss':test_loss,
            'train_loss_curve':train_loss_curve, 'val_loss_curve':test_loss_curve, 'fit_time':fit_time,
            'hp_idx':hp_idx, 'split_idx':split_idx, 'seed':seed, "n_splits":n_splits,
            **hps,
            **special_hps_for_gs
        }

        # Save gs res
        with open(f"{gs_res_dir}/gs_res_{gs_name}_{dataset_name}_{seed}_seed_{split_idx}_split_idx_{hp_idx}_hp_idx.json", 'w') as f:
            json.dump(gs_res, f)

        # Save gs_models (e.g., cv of best model)
        if save_gs_models:
            with open(f"{trained_models_dir}/{timestamp}_model_{gs_name}_{dataset_name}_seed_{seed}_{n_splits}fold_cv__hp_idx_{hp_idx}_split_idx_{split_idx}.pkl", 'wb') as f:
                pickle.dump(model, f)
 
    # Save model from single fit
    else:
        with open(f"{trained_models_dir}/{timestamp}_model_single_fit_{dataset_name}_seed_{seed}.pkl", 'wb') as f:
            pickle.dump(model, f)

    print('done')