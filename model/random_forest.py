# Random Forest (Ensemble) model for ML Assignment 2
# run:  python model/random_forest.py          (uses the best params I found)
#       python model/random_forest.py --tune   (runs the grid search again)

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from data_preprocessing import RANDOM_STATE, build_preprocessor, get_train_test
from evaluation import train_and_evaluate

MODEL_NAME = "Random Forest (Ensemble)"

# params tried in GridSearchCV (5 fold, on train data only)
PARAM_GRID = {
    "clf__max_depth": [None, 15, 25],
    "clf__min_samples_leaf": [1, 3, 5],
}

# best params from my grid search run (see results/grid_search_log.txt)
BEST_PARAMS = {"clf__max_depth": 25, "clf__min_samples_leaf": 3}


def build_pipeline():
    # random forest = bagging of decision trees, no scaling needed
    # preprocessing is inside the pipeline so it only gets fitted on train data
    return Pipeline([
        ("prep", build_preprocessor(scale_numeric=False, dense_output=False)),
        ("clf", RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=-1,
                                      random_state=RANDOM_STATE)),
    ])


def run(X_train, X_test, y_train, y_test, tune=False):
    return train_and_evaluate(MODEL_NAME, build_pipeline(), PARAM_GRID, BEST_PARAMS,
                              X_train, X_test, y_train, y_test, tune=tune)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    run(*get_train_test(), tune=args.tune)
