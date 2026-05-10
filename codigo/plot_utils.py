import os
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_history(histories, title, file_path):
    if not histories:
        return

    max_epochs = max(len(h.history.get("loss", [])) for h in histories)
    if max_epochs == 0:
        return

    epochs = np.arange(1, max_epochs + 1)
    avg_loss = _average_metric(histories, "loss", max_epochs)
    avg_val_loss = _average_metric(histories, "val_loss", max_epochs)
    avg_acc = _average_metric(histories, "accuracy", max_epochs)
    avg_val_acc = _average_metric(histories, "val_accuracy", max_epochs)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, avg_loss, label="train")
    plt.plot(epochs, avg_val_loss, label="val")
    plt.title(f"Loss - {title}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, avg_acc, label="train")
    plt.plot(epochs, avg_val_acc, label="val")
    plt.title(f"Accuracy - {title}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()


def _average_metric(histories, key, max_epochs):
    matrix = np.full((len(histories), max_epochs), np.nan)
    for i, h in enumerate(histories):
        values = h.history.get(key, [])
        if values:
            matrix[i, :len(values)] = values
    return np.nanmean(matrix, axis=0)
