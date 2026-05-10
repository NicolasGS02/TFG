import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def list_csv_files(data_path):
    return [
        os.path.join(data_path, name)
        for name in os.listdir(data_path)
        if name.lower().endswith(".csv")
    ]


def load_dataset(file_path):
    df = pd.read_csv(file_path)
    target_col = _find_target_column(df)
    y = df[target_col]
    X = df.drop(columns=[target_col])

    X = _encode_features(X)
    y = _encode_labels(y)

    return X.to_numpy(), y


def _find_target_column(df):
    for name in df.columns:
        if str(name).strip().lower() == "class":
            return name
    return df.columns[-1]


def _encode_features(X):
    obj_cols = X.select_dtypes(include=["object", "category"]).columns
    if len(obj_cols) > 0:
        X = pd.get_dummies(X, columns=obj_cols)
    return X


def _encode_labels(y):
    encoder = LabelEncoder()
    return encoder.fit_transform(y.to_numpy().ravel())
