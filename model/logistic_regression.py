# Logistic Regression model for ML Assignment 2
# run:  python model/logistic_regression.py          (uses the best params I found)
#       python model/logistic_regression.py --tune   (runs the grid search again)

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data_preprocessing import RANDOM_STATE, build_preprocessor, get_train_test
from evaluation import train_and_evaluate

MODEL_NAME = "Logistic Regression"

# params tried in GridSearchCV (5 fold, on train data only)
PARAM_GRID = {"clf__C": [0.01, 0.1, 1.0, 10.0]}

# best params from my grid search run (see results/grid_search_log.txt)
BEST_PARAMS = {"clf__C": 0.1}


def build_pipeline():
    # logistic regression needs scaled inputs, so scaling is on
    # preprocessing is inside the pipeline so it only gets fitted on train data
    return Pipeline([
        ("prep", build_preprocessor(scale_numeric=True, dense_output=False)),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE)),
    ])


def run(X_train, X_test, y_train, y_test, tune=False):
    return train_and_evaluate(MODEL_NAME, build_pipeline(), PARAM_GRID, BEST_PARAMS,
                              X_train, X_test, y_train, y_test, tune=tune)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    run(*get_train_test(), tune=args.tune)
