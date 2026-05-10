import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def summarize_metrics(metrics_list):
    summary = {}
    if not metrics_list:
        return summary

    keys = metrics_list[0].keys()
    for key in keys:
        summary[key] = float(np.mean([m[key] for m in metrics_list]))
    return summary
