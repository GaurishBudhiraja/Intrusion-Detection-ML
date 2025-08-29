# visualize.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -------------------------
# 1) Training curves
# -------------------------
def plot_training_curves(history, metrics=("loss",), title="Training Curves"):
    """
    Plot training and validation curves from Keras history object.
    """
    plt.figure(figsize=(8, 5))
    for m in metrics:
        if m in history.history:
            plt.plot(history.history[m], label=f"train_{m}")
        if f"val_{m}" in history.history:
            plt.plot(history.history[f"val_{m}"], label=f"val_{m}")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------
# 2) Reconstruction errors
# -------------------------
def plot_recon_error_distribution(err_normal, err_fault, thr=None):
    """
    Plot reconstruction error histograms for normal vs. fault data.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(err_normal, bins=50, alpha=0.6, label="Normal")
    plt.hist(err_fault, bins=50, alpha=0.6, label="Fault")
    if thr is not None:
        plt.axvline(thr, color="red", linestyle="--", label=f"Threshold {thr:.4f}")
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# -------------------------
# 3) Confusion Matrix
# -------------------------
def plot_confusion(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    """
    Plot confusion matrix for classification.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# -------------------------
# 4) Regression scatter
# -------------------------
def plot_regression_scatter(y_true, y_pred, title="Localization Scatter"):
    """
    Scatter plot for regression: true vs predicted values.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolor="k")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", label="Ideal")
    plt.xlabel("True Position")
    plt.ylabel("Predicted Position")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
