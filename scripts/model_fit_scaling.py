if __name__ == '__main__':

    from sklearn.datasets import make_multilabel_classification
    from sklearn.ensemble import RandomForestClassifier
    import time
    import numpy as np

    rng = np.random.default_rng(seed=1234)

    # Toy datasets
    ns = 250
    nf = 50
    n_classes = [2, 10, 100, 1000]

    for nc in n_classes:
        X, y = make_multilabel_classification(n_samples=ns, n_classes=nc, n_features=nf, random_state=0)
        clf = RandomForestClassifier(n_jobs=1, max_depth=7, max_samples=int(np.sqrt(ns)), n_estimators=1, max_features='sqrt')
        tic = time.perf_counter()
        clf.fit(X, y)
        toc = time.perf_counter()
        ft = toc - tic
        print(f"{nc} classes {ft} seconds")

    print('done')