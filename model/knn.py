# kNN model for ML Assignment 2
# run:  python model/knn.py          (uses the best params I found)
#       python model/knn.py --tune   (runs the grid search again)

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from data_preprocessing import RANDOM_STATE, build_preprocessor, get_train_test
from evaluation import train_and_evaluate

MODEL_NAME = "kNN"

# params tried in GridSearchCV (5 fold, on train data only)
PARAM_GRID = {
    "clf__n_neighbors": [5, 11, 21, 31, 41],
    "clf__weights": ["uniform", "distance"],
}

# best params from my grid search run (see results/grid_search_log.txt)
BEST_PARAMS = {"clf__n_neighbors": 41, "clf__weights": "uniform"}


def build_pipeline():
    # kNN uses distances, so scaling is a must (otherwise fnlwgt dominates everything)
    # preprocessing is inside the pipeline so it only gets fitted on train data
    return Pipeline([
        ("prep", build_preprocessor(scale_numeric=True, dense_output=False)),
        ("clf", KNeighborsClassifier(metric="minkowski", p=2)),
    ])


def run(X_train, X_test, y_train, y_test, tune=False):
    return train_and_evaluate(MODEL_NAME, build_pipeline(), PARAM_GRID, BEST_PARAMS,
                              X_train, X_test, y_train, y_test, tune=tune)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    run(*get_train_test(), tune=args.tune)
