# Naive Bayes model for ML Assignment 2
# run:  python model/naive_bayes.py          (uses the best params I found)
#       python model/naive_bayes.py --tune   (runs the grid search again)

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from data_preprocessing import RANDOM_STATE, build_preprocessor, get_train_test
from evaluation import train_and_evaluate

MODEL_NAME = "Naive Bayes"

# params tried in GridSearchCV (5 fold, on train data only)
PARAM_GRID = {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 0.3]}

# best params from my grid search run (see results/grid_search_log.txt)
BEST_PARAMS = {"clf__var_smoothing": 0.01}


def build_pipeline():
    # using GaussianNB since we have continuous + one-hot columns (MultinomialNB needs counts).
    # GaussianNB doesn't accept sparse input, hence dense_output=True
    # preprocessing is inside the pipeline so it only gets fitted on train data
    return Pipeline([
        ("prep", build_preprocessor(scale_numeric=True, dense_output=True)),
        ("clf", GaussianNB()),
    ])


def run(X_train, X_test, y_train, y_test, tune=False):
    return train_and_evaluate(MODEL_NAME, build_pipeline(), PARAM_GRID, BEST_PARAMS,
                              X_train, X_test, y_train, y_test, tune=tune)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    run(*get_train_test(), tune=args.tune)
