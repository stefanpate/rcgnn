from argparse import ArgumentParser
from src.config import filepaths
from src.task import construct_featurizer, construct_model


def _predict_cv(args):
    for i, j in zip(foo, bar):
            chkpt = torch.load(filepath, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            model.load_state_dict(chkpt['state_dict'])

parser = ArgumentParser(description="Predict score on binary protein-reaction binary classification")
subparsers = parser.add_subparsers(title="Commands", description="Available commands")

# Predict with series of models from a cross validation
predict_cv = subparsers.add_parser("predict-cv", description="Predicts using series of models of same hyperparams from a kfold cross validation")
predict_cv.add_argument("hp-idx", help="Hyperparameter index in experiments csv")
predict_cv.set_defaults(func=_predict_cv)

def main():
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()