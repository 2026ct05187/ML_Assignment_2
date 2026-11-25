# Main script - runs the whole assignment:
#   dataset summary -> preprocessing -> train/test split -> 5 models -> metrics table -> plots
#
#   python model/train_all.py          (fast, uses tuned params, ~1 min)
#   python model/train_all.py --tune   (re-runs grid search for all models, ~10-15 min)

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

import decision_tree
import knn
import logistic_regression
import naive_bayes
import random_forest
from data_preprocessing import (
    PROJECT_DIR, TARGET, clean_data, dataset_summary, load_raw_data,
    save_test_data, split_data,
)
from evaluation import METRIC_NAMES, metrics_table

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# colours for the 5 models (same colour per model in every chart)
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def _blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blues", BLUES)


def plot_class_distribution(df):
    counts = df[TARGET].value_counts()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    bars = ax.bar(counts.index, counts.values, color=MODEL_COLORS[0], width=0.55)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / counts.sum():.1%})",
                ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_title("Class distribution of target 'income'")
    ax.set_ylabel("Number of instances")
    ax.set_ylim(0, counts.max() * 1.22)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=150)
    plt.close(fig)


def plot_missing_values(df):
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values()
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.barh(miss.index, miss.values, color=MODEL_COLORS[1], height=0.5)
    for i, v in enumerate(miss.values):
        ax.text(v, i, f"  {v:,} ({v / len(df):.1%})", va="center", color=INK, fontsize=9)
    ax.set_title("Missing values per feature (raw data)")
    ax.set_xlabel("Number of missing values ('?')")
    ax.set_xlim(0, miss.max() * 1.35)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "missing_values.png"), dpi=150)
    plt.close(fig)


def plot_metric_comparison(table):
    models = list(table.index)
    x = np.arange(len(METRIC_NAMES))
    w = 0.16
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, (m, c) in enumerate(zip(models, MODEL_COLORS)):
        vals = table.loc[m, METRIC_NAMES].values
        pos = x + (i - (len(models) - 1) / 2) * w
        ax.bar(pos, vals, width=w - 0.02, color=c, label=m)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.008, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6.5, color=INK_2, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_NAMES)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (test set)")
    ax.set_title("Evaluation metrics of all 5 models (test set, 20%)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "metrics_comparison_bar.png"), dpi=150)
    plt.close(fig)


def plot_metric_heatmap(table):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    data = table[METRIC_NAMES].values
    im = ax.imshow(data, cmap=_blue_cmap(), vmin=data.min() - 0.05, vmax=1.0,
                   aspect="auto")
    ax.set_xticks(range(len(METRIC_NAMES)))
    ax.set_xticklabels(METRIC_NAMES)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    thresh = data.min() + (1.0 - data.min()) * 0.55
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.4f}", ha="center", va="center", fontsize=9,
                    color="white" if data[i, j] > thresh else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Metric heatmap (darker = better)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "metrics_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for r, c in zip(results, MODEL_COLORS):
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        ax.plot(fpr, tpr, color=c, linewidth=2,
                label=f"{r['name']} (AUC = {r['metrics']['AUC']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=INK_2, linewidth=1,
            label="Random (AUC = 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (Recall)")
    ax.set_title("ROC curves (test set)")
    ax.grid(color=GRID, linewidth=0.8)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"), dpi=150)
    plt.close(fig)


def plot_confusion_matrices(results, y_test):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6))
    axes = axes.ravel()
    axes[-1].axis("off")
    for ax, r in zip(axes, results):
        cm = confusion_matrix(y_test, r["y_pred"])
        ax.imshow(cm, cmap=_blue_cmap())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() * 0.55 else INK,
                        fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["<=50K", ">50K"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["<=50K", ">50K"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(r["name"], fontsize=10.5)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Confusion matrices (test set)", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "confusion_matrices.png"), dpi=150)
    plt.close(fig)


def main(tune=False):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # step 1 - dataset
    df = load_raw_data()
    summary = dataset_summary(df)
    print("\n### DATASET: UCI Adult (Census Income) ###")
    print(f"Instances : {summary['n_instances']:,}")
    print(f"Features  : {summary['n_features']} "
          f"({len(summary['numeric_features'])} numeric, "
          f"{len(summary['categorical_features'])} categorical)")
    print("Numeric     :", summary["numeric_features"])
    print("Categorical :", summary["categorical_features"])
    print("Missing values per column:", summary["missing_per_column"])
    print(f"Rows with >=1 missing value: {summary['rows_with_missing']:,}")
    print(f"Exact duplicate rows: {summary['n_duplicates']:,}")
    print("Class counts:", summary["class_counts"])
    print("Class %     :", summary["class_pct"])
    plot_class_distribution(df)
    plot_missing_values(df)

    # step 2 - preprocessing + split
    X, y, n_dupes = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"\nDuplicates removed: {n_dupes:,}  ->  {len(X):,} rows remain")
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows (80:20, stratified)")
    print(f"Positive (>50K) share  train: {y_train.mean():.4f} | test: {y_test.mean():.4f}")
    save_test_data(X_test, y_test)
    print("Saved test split -> test_data.csv")

    # step 3 - train and evaluate all 5 models
    modules = [logistic_regression, decision_tree, knn, naive_bayes, random_forest]
    results = [m.run(X_train, X_test, y_train, y_test, tune=tune) for m in modules]

    table = metrics_table(results)
    print("\n### COMPARISON TABLE (test set) ###")
    print(table.round(4).to_string())
    table.round(4).to_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"))
    with open(os.path.join(RESULTS_DIR, "metrics_comparison.md"), "w") as f:
        f.write(table.round(4).to_markdown())

    # save some extra details for the report
    extra = {
        "dataset": {k: v for k, v in summary.items()},
        "duplicates_removed": n_dupes,
        "rows_after_cleaning": len(X),
        "train_rows": len(X_train), "test_rows": len(X_test),
        "train_pos_rate": float(y_train.mean()), "test_pos_rate": float(y_test.mean()),
        "models": {r["name"]: {
            "params": {k: (v if isinstance(v, (int, float, str)) or v is None else str(v))
                       for k, v in r["params"].items()},
            "cv_auc": r["cv_auc"], "train_time_s": round(r["train_time"], 2),
            "confusion_matrix": confusion_matrix(y_test, r["y_pred"]).tolist(),
            "metrics": {k: round(float(v), 4) for k, v in r["metrics"].items()},
        } for r in results},
    }
    with open(os.path.join(RESULTS_DIR, "results_summary.json"), "w") as f:
        json.dump(extra, f, indent=2, default=str)

    plot_metric_comparison(table)
    plot_metric_heatmap(table)
    plot_roc_curves(results, y_test)
    plot_confusion_matrices(results, y_test)
    print(f"\nPlots saved in {PLOTS_DIR}")
    print(f"Models saved in {os.path.join(PROJECT_DIR, 'model', 'saved_models')}")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true",
                        help="re-run 5-fold GridSearchCV for every model")
    main(tune=parser.parse_args().tune)
