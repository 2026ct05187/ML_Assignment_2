# Loads the saved .pkl models and checks them again on test_data.csv
# (should give the same numbers as train_all.py)
# run train_all.py once before this

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import pandas as pd

from data_preprocessing import PROJECT_DIR
from evaluation import METRIC_NAMES, SAVED_MODELS_DIR, compute_metrics

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}


def main():
    test = pd.read_csv(os.path.join(PROJECT_DIR, "test_data.csv"))
    y_test = (test.pop("income") == ">50K").astype(int)
    rows = []
    for name, fname in MODEL_FILES.items():
        model = joblib.load(os.path.join(SAVED_MODELS_DIR, fname))
        y_pred = model.predict(test)
        y_proba = model.predict_proba(test)[:, 1]
        rows.append({"ML Model Name": name, **compute_metrics(y_test, y_pred, y_proba)})
    table = pd.DataFrame(rows).set_index("ML Model Name")[METRIC_NAMES]
    print(f"Test rows: {len(test):,}")
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
