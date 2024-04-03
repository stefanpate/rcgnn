import multiprocessing as mp
import queue
# from multiprocessing import set_start_method
from collections import namedtuple
from sklearn.tree import DecisionTreeClassifier
import pickle

TreeParamSet = namedtuple('TreeParamSet', 'max_depth, max_samples, max_features')
MetaData = namedtuple('MetaData', 'outer_split, inner_split, bootstrap_idx, hp_idx', defaults=[None for _ in range(4)])

def fit_split(q_in, q_out, lock):
    while True:
        # try:
        elt = q_in.get()
        if elt is None:
            break
        model, X, y, meta_data = elt
        # except queue.Empty:
            # break
        with lock:
            print(f"Fitting {meta_data}")
        model.fit(X, y)
        
        fn = f"dt_os_{meta_data.outer_split}_is_{meta_data.inner_split}_bs_{meta_data.bootstrap_idx}_hp_{meta_data.hp_idx}.pkl"
        with open(f"/scratch/spn1560/decision_trees/{fn}", 'wb') as f:
            pickle.dump(model, f)
        q_out.put((fn, meta_data))


if __name__ == '__main__':

    from sklearn.model_selection import KFold
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
    from itertools import product
    from src.utils import load_sparse_adj_mat, load_design_matrix
    import numpy as np
    from src.utils import ensure_dirs

    ensure_dirs("/scratch/spn1560/decision_trees")

    # Settings
    train_data_name = 'swissprot'
    embed_type = 'esm'
    n_cores = 2
    inner_kfold = 2
    outer_kfold = 2
    hpo_metric = 'f1_weighted'
    seed = 1234
    do_save = True
    rng = np.random.default_rng(seed=seed)

    # Load dataset
    y, idx_sample, idx_feature = load_sparse_adj_mat(train_data_name)
    sample_idx = {v:k for k,v in idx_sample.items()}
    X = load_design_matrix(train_data_name, embed_type, sample_idx, do_norm=True)[:1000, :100]
    y = y[:1000, :100].toarray() # Use dense array for now... don't know why sklearn complaining about sparse array...

    # Params for grid search
    max_depths = [3] # [3, 5, 7]
    n_estimators = [2] # [50, 75, 100]
    max_samples = [int(np.sqrt(X.shape[0]))]
    max_features = [int(np.sqrt(X.shape[1]))]
    tree_params = [TreeParamSet(*elt) for elt in product(max_depths, max_samples, max_features)]
    T = max(n_estimators)

    # Splits for nested cv
    inner_cv = KFold(n_splits=inner_kfold, shuffle=True, random_state=seed)
    outer_cv = KFold(n_splits=outer_kfold, shuffle=True, random_state=seed)

    # Initialize queues and pool
    to_fit = mp.Queue(maxsize=n_cores)
    fitted = mp.Queue()
    the_lock = mp.Lock()
    pool = mp.Pool(n_cores, initializer=fit_split, initargs=(to_fit, fitted, the_lock))

    # Populate the in queue
    for o, (train_val_idxs, test_idxs) in enumerate(outer_cv.split(X)):
        X_train_val, y_train_val = X[train_val_idxs], y[train_val_idxs]
        X_test, y_test = X[test_idxs], y[test_idxs]
        for i, (train_idxs, val_idxs) in enumerate(inner_cv.split(X_train_val)):
            X_train, y_train = X_train_val[train_idxs], y_train_val[train_idxs]
            X_val, y_val = X_train_val[val_idxs], y_train_val[val_idxs]
            for h, tree_param in enumerate(tree_params):
                for b in range(T):
                    bootstrap_idxs = rng.choice(np.arange(X_train.shape[0]), size=(tree_param.max_samples, ), replace=False)
                    model = DecisionTreeClassifier(max_depth=tree_param.max_depth, max_features=tree_param.max_features, random_state=seed)
                    X_bs, y_bs = X_train[bootstrap_idxs], y_train[bootstrap_idxs]
                    meta_data = MetaData(o, i, b, h)
                    elt = (model, X_bs, y_bs, meta_data)
                    # to_fit.put(elt)
                    to_fit.put((model, X_bs, y_bs, meta_data))

                    with the_lock:
                        print(f"Queued: {meta_data}")

    # Signal to workers: 'all done'
    for _ in range(n_cores):
        to_fit.put(None)

    pool.close()
    pool.join()

    while not fitted.empty():
        elt = fitted.get()
        fn = elt[0]
        break

    # model = pickle.loads(model_s)
    # model.predict(X_bs)
    with open(f"/scratch/spn1560/decision_trees/{fn}", 'rb') as f:
        model = pickle.load(f)
    print('check')
