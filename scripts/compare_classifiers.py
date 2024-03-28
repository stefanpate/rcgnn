from sklearn.model_selection import GridSearchCV, KFold
from collections import defaultdict


def nested_cv(X, y, model, param_grid, hpo_metric, scoring_metrics, inner_k, outer_k):

    # Perform nested cross-validation
    inner_cv = KFold(n_splits=inner_k, shuffle=True, random_state=42)
    outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=42)

    outer_scores = defaultdict(list)
    best_hps = None
    best_score = 0

    for train_index, test_index in outer_cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=hpo_metric, cv=inner_cv)
        grid_search.fit(X_train, y_train)
        inner_best_estimator = grid_search.best_estimator_ # Take this to generate outer score (estimate generalization score)

        # Save best parameters overall for model selection
        if grid_search.best_score_ >= best_score:
            best_hps = grid_search.best_params_

        # Estimate generalization score
        inner_best_estimator.fit(X_train, y_train)
        y_pred = inner_best_estimator.predict(X_test)

        for metric_name, metric in scoring_metrics.items():
            outer_scores[metric_name].append(metric(y_test, y_pred))

    return outer_scores, best_hps


if __name__ == '__main__':

    from sklearn.datasets import load_iris, make_multilabel_classification
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
    from itertools import product
    import pickle
    import json

    inner_kfold = 5
    outer_kfold = 5
    hpo_metric = 'f1_weighted'
    do_save = True
    train_data_name = 'foo'
    embeds_name = 'bar'

    names = [
        "svm",
        "random_forest",
        "naive_bayes",
        "logistic_regression"
    ]

    classifiers = [
        MultiOutputClassifier(SVC()),
        RandomForestClassifier(),
        MultiOutputClassifier(GaussianNB()),
        MultiOutputClassifier(LogisticRegression())
    ]

    # Construct svm estimators of varied params for MultiOutputClassifier
    # wrapper during grid search
    svm_params = {'kernel' : ['linear', 'rbf'], 'C' : [0.1, 1]}
    svm_param_combos = list(product(*svm_params.values())) # Cartesian product of svm parameters
    svm_estimators = [SVC(**{k:param_combo[i] for i, k in enumerate(svm_params.keys())}) for param_combo in svm_param_combos] # Construct svm estimators

    parameter_grids = [
        {'estimator' : svm_estimators},
        {'n_estimators' : [10, 100]},
        {'estimator' : [GaussianNB()]},
        {'estimator' : [LogisticRegression()]}
    ]

    scoring_metrics = {
        'accuracy': accuracy_score,
        'f1_weighted' : lambda y, y_pred : f1_score(y, y_pred, average='weighted'),
        'recall_weighted' : lambda y, y_pred : recall_score(y, y_pred, average='weighted'),
        'precision_weighted' : lambda y, y_pred : precision_score(y, y_pred, average='weighted'),
        'roc_auc_weighted' : lambda y, y_pred : roc_auc_score(y, y_pred, average='weighted'),
        'roc_auc_macro' : lambda y, y_pred : roc_auc_score(y, y_pred, average='macro'),
        'f1_samples' : lambda y, y_pred : f1_score(y, y_pred, average='samples', zero_division=0),
        'recall_samples' : lambda y, y_pred : recall_score(y, y_pred, average='samples', zero_division=0),
        'precision_samples' : lambda y, y_pred : precision_score(y, y_pred, average='samples', zero_division=0)
    }
    
    # Load dataset
    X, y = make_multilabel_classification(n_classes=3, random_state=0)
    # Xi, yi = load_iris(return_X_y=True)
 
    for name, model, param_grid in zip(names, classifiers, parameter_grids):
        print(f"Tuning: {name}")

        generalization_scores, best_hps = nested_cv(X, y, model, param_grid, hpo_metric, scoring_metrics, inner_kfold, outer_kfold)
        print(generalization_scores)

        # Fit model w/ best HPs on whole training data
        production_model = model.set_params(**best_hps)
        production_model.fit(X, y)

        # Save production model & generalization scores for this model
        if do_save:
            prefix = f"model_{name}_train_data_{train_data_name}_protein_embeddings_{embeds_name}_function_embeddings_EC"

            with open(f"../artifacts/trained_models/{prefix}.pkl", 'wb') as f:
                pickle.dump(production_model, f)

            with open(f"../artifacts/model_evals/{prefix}.json", 'w') as f:
                json.dump(generalization_scores, f)

            



    print('done')