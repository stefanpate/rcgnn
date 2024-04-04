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

    n_features = 1024
    n_classes = 5242
    n_estimators = [1,]
    max_samples = [int(np.sqrt(228000)),]
    mf = int(np.sqrt(5242))

  
    # Load dataset
    # y, idx_sample, idx_feature = load_sparse_adj_mat(train_data_name)
    # sample_idx = {v:k for k,v in idx_sample.items()}
    # X = load_design_matrix(train_data_name, embed_type, sample_idx, do_norm=True)
    # print(getsizeof(X))
    # y = y.toarray() # Use dense array for now... don't know why sklearn complaining about sparse array...

    # # Toy datasets
    ns = 5000
    X, y = make_multilabel_classification(n_samples=ns, n_classes=5242, n_features=1024, random_state=0)

    for ms in max_samples:
        for ne in n_estimators:
            clf = RandomForestClassifier(n_jobs=1, max_depth=7, max_samples=ms, n_estimators=ne, max_features=mf)
            tic = time.perf_counter()
            clf.fit(X, y)
            toc = time.perf_counter()
            ft = toc - tic
            print(f"{ne} trees trained on {ms} samples & {mf} features in {ft} seconds")

    print('done')