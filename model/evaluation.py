# Common code for training, tuning and evaluating the models.
# Metrics: Accuracy, AUC, Precision, Recall, F1, MCC  (positive class = >50K)

import os
import time

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from data_preprocessing import RANDOM_STATE, PROJECT_DIR

SAVED_MODELS_DIR = os.path.join(PROJECT_DIR, "model", "saved_models")
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "AUC":       roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred),
        "F1":        f1_score(y_true, y_pred),
        "MCC":       matthews_corrcoef(y_true, y_pred),
    }


def train_and_evaluate(name, pipeline, param_grid, default_params,
                       X_train, X_test, y_train, y_test, tune=False,
                       save=True, verbose=True):
    # tune=True  -> GridSearchCV (5 fold, roc_auc) on train data
    # tune=False -> just fit with the best params found earlier
    # either way the test set is only used at the end for scoring
    start = time.time()
    if tune:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        search = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=cv,
                              n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        cv_auc = search.best_score_
    else:
        model = pipeline.set_params(**default_params)
        model.fit(X_train, y_train)
        best_params = default_params
        cv_auc = None
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    if verbose:
        print("=" * 70)
        print(f"{name}")
        print("=" * 70)
        print("Parameters:", best_params)
        if cv_auc is not None:
            print(f"Best 5-fold CV ROC-AUC (train): {cv_auc:.4f}")
        print(f"Training time: {train_time:.1f}s")
        for k, v in metrics.items():
            print(f"  {k:<10}: {v:.4f}")
        print("Confusion matrix [[TN FP] [FN TP]]:")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred,
                                    target_names=["<=50K", ">50K"], digits=4))

    if save:
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, os.path.join(SAVED_MODELS_DIR, f"{fname}.pkl"),
                    compress=3)

    return {
        "name": name, "model": model, "metrics": metrics,
        "params": best_params, "cv_auc": cv_auc,
        "y_pred": y_pred, "y_proba": y_proba, "train_time": train_time,
    }


def metrics_table(results):
    rows = [{"ML Model Name": r["name"], **r["metrics"]} for r in results]
    return pd.DataFrame(rows).set_index("ML Model Name")[METRIC_NAMES]
