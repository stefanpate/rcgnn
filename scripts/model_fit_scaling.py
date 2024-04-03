if __name__ == '__main__':

    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.datasets import make_multilabel_classification
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from src.utils import load_sparse_adj_mat, load_design_matrix
    import time
    import numpy as np
    from sys import getsizeof

    rng = np.random.default_rng(seed=1234)

    # Settings
    train_data_name = 'swissprot'
    embed_type = 'esm'

    # processes = [10]
    # n_samples = [1e1, 1e2, 1e3]
    n_features = 1000
    n_classes = 5000
    n_estimators = [1,]
    max_samples = [int(np.sqrt(228000)),]
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    mf = int(np.sqrt(5000))

  
    # Load dataset
    y, idx_sample, idx_feature = load_sparse_adj_mat(train_data_name)
    sample_idx = {v:k for k,v in idx_sample.items()}
    X = load_design_matrix(train_data_name, embed_type, sample_idx, do_norm=True)
    print(getsizeof(X))
    # y = y.toarray() # Use dense array for now... don't know why sklearn complaining about sparse array...

    # # Toy datasets
    # X, y = make_multilabel_classification(n_classes=3, random_state=0)
    
    ns = 1000
    X = rng.normal(size=(int(ns), n_features))
    y = (rng.uniform(size=(int(ns), n_classes)) > 0.7).astype(int)

    for ms in max_samples:
        for ne in n_estimators:
            clf = RandomForestClassifier(n_jobs=1, max_depth=7, max_samples=ms, n_estimators=ne, max_features=mf)
            # grid_search = GridSearchCV(estimator=clf, param_grid={'max_depth':[3], 'max_samples': [ms], 'n_estimators': [ne]}, cv=inner_cv, n_jobs=10)
            tic = time.perf_counter()
            # grid_search.fit(X, y)
            clf.fit(X, y)
            toc = time.perf_counter()
            ft = toc - tic
            print(f"{ne} trees trained on {ms} samples & {mf} features in {ft} seconds")

    print('done')