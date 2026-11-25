# Data loading + preprocessing for the Adult dataset.
# All 5 models import from here so they all use the exact same train/test split.
#
# Steps done (as per assignment step 2):
#   - missing values: '?' -> NaN, then imputed (most frequent / median)
#   - duplicates: dropped
#   - encoding: one-hot for categorical columns
#   - scaling: StandardScaler, only for models that need it
#   - 80:20 split, stratified, random_state = 42
#   - imputer/encoder/scaler are inside a Pipeline so they are fit on train data only

import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.20

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]
TARGET = "income"

NUMERIC_FEATURES = [
    "age", "fnlwgt", "education-num", "capital-gain",
    "capital-loss", "hours-per-week",
]
CATEGORICAL_FEATURES = [
    "workclass", "marital-status", "occupation", "relationship",
    "race", "sex", "native-country",
]
# education and education-num carry the same info (text vs number), so I drop the text one
DROPPED_FEATURES = ["education"]


def _ensure_raw_files():
    # use the files in data/ if they are there, otherwise unzip / download from UCI
    data_path = os.path.join(DATA_DIR, "adult.data")
    test_path = os.path.join(DATA_DIR, "adult.test")
    if os.path.exists(data_path) and os.path.exists(test_path):
        return data_path, test_path

    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "adult.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading dataset from {UCI_ZIP_URL} ...")
        urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    return data_path, test_path


def load_raw_data():
    # adult.data (32561 rows) + adult.test (16281 rows) = 48842 rows
    # missing values are stored as '?' in the files
    data_path, test_path = _ensure_raw_files()
    read_kw = dict(header=None, names=COLUMNS, sep=",", skipinitialspace=True,
                   na_values="?", keep_default_na=False)
    df_a = pd.read_csv(data_path, **read_kw)
    df_b = pd.read_csv(test_path, skiprows=1, **read_kw)   # first line of adult.test is junk
    df = pd.concat([df_a, df_b], ignore_index=True)

    # labels in adult.test have a trailing '.' (">50K.") so remove it
    df[TARGET] = df[TARGET].astype(str).str.strip().str.rstrip(".")
    df = df[df[TARGET].isin(["<=50K", ">50K"])].reset_index(drop=True)
    return df


def dataset_summary(df):
    # basic stats I used for the dataset description
    feats = [c for c in df.columns if c != TARGET]
    return {
        "n_instances": len(df),
        "n_features": len(feats),
        "numeric_features": [c for c in feats if c in NUMERIC_FEATURES],
        "categorical_features": [c for c in feats if c not in NUMERIC_FEATURES],
        "missing_per_column": df.isna().sum()[lambda s: s > 0].to_dict(),
        "rows_with_missing": int(df.isna().any(axis=1).sum()),
        "n_duplicates": int(df.duplicated().sum()),
        "class_counts": df[TARGET].value_counts().to_dict(),
        "class_pct": (df[TARGET].value_counts(normalize=True) * 100).round(2).to_dict(),
    }


def clean_data(df):
    # remove duplicates, drop 'education', make target 0/1
    # (imputation is NOT done here, it happens in the pipeline after the split)
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dupes = n_before - len(df)

    df = df.drop(columns=DROPPED_FEATURES)
    y = (df[TARGET] == ">50K").astype(int)      # 1 = >50K
    X = df.drop(columns=[TARGET])
    return X, y, n_dupes


def split_data(X, y):
    # stratify so both parts keep the same ~76:24 class ratio
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                            random_state=RANDOM_STATE)


def build_preprocessor(scale_numeric=True, dense_output=False):
    # numeric: median impute (+ scaling if needed)
    # categorical: most frequent impute + one hot
    # trees use scale_numeric=False, GaussianNB needs dense_output=True
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore",
                                 sparse_output=not dense_output)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), NUMERIC_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.0 if dense_output else 0.3,
    )


def get_train_test():
    # load -> clean -> split, used by every model script
    df = load_raw_data()
    X, y, _ = clean_data(df)
    return split_data(X, y)


def save_test_data(X_test, y_test, path=None):
    # save the test split (20%) to test_data.csv
    path = path or os.path.join(PROJECT_DIR, "test_data.csv")
    out = X_test.copy()
    out["income"] = np.where(y_test == 1, ">50K", "<=50K")
    out.to_csv(path, index=False)
    return path
