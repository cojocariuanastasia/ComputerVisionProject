"""
Loads the trained gesture classifier and the dataset CSV,
runs evaluation on a held-out test split, and saves plots to results/.

Usage:
    python evaluate.py
    python evaluate.py --data data/gestures_dataset.csv --model models/gesture_rf.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split


RESULTS_DIR = Path("results")
RANDOM_STATE = 42
TEST_SIZE = 0.2
FOLDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  type=str, default="data/gestures_dataset.csv")
    parser.add_argument("--model", type=str, default="models/gesture_rf.joblib")
    return parser.parse_args()


def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    x = df.drop(columns=["label"])
    y = df["label"]
    return x, y, train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def plot_confusion_matrix(y_test, y_pred, labels: list, out: Path) -> None:
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_pct, annot=False, fmt=".1f", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=0.5, linecolor="white",
    )

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j + 0.5, i + 0.5,
                    f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=10,
                    color="white" if cm_pct[i, j] > 50 else "black")

    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved to {out}")


def plot_class_metrics(y_test, y_pred, labels: list, out: Path) -> None:
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True)

    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, metric in enumerate(metrics):
        values = [report[label][metric] for label in labels]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize(),
                      edgecolor="white")
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-class Precision / Recall / F1-score", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(y=report["accuracy"], color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(len(labels) - 0.1, report["accuracy"] + 0.01,
            f"Accuracy: {report['accuracy']:.3f}", color="red", fontsize=9, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved to {out}")


def plot_dataset_distribution(y_full: pd.Series, labels: list, out: Path) -> None:
    counts = [int((y_full == label).sum()) for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    bars = axes[0].bar(labels, counts, edgecolor="white")
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     str(count), ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Number of samples", fontsize=12)
    axes[0].set_title("Samples per class", fontsize=13, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    wedges, _, autotexts = axes[1].pie(
        counts, labels=labels,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(10)
    axes[1].set_title(f"Class distribution  (total: {sum(counts)})",
                      fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved to {out}")


def plot_cross_validation(model, x: pd.DataFrame, y: pd.Series, out: Path) -> dict:
    """
    Runs StratifiedKFold CV and plots accuracy per fold + mean/std band.
    Stratified ensures each fold has the same class distribution.
    Returns the cv_results dict for the text report.
    """
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    cv_results = cross_validate(model, x, y, cv=cv, scoring=scoring, n_jobs=-1)

    fold_accs = cv_results["test_accuracy"]
    folds     = np.arange(1, FOLDS + 1)
    mean_acc  = fold_accs.mean()
    std_acc   = fold_accs.std()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: accuracy per fold with mean and std band
    bars = axes[0].bar(folds, fold_accs, edgecolor="white", alpha=0.85, width=0.5)
    for bar, val in zip(bars, fold_accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    axes[0].axhline(mean_acc, color="red", linestyle="--", linewidth=1.5,
                    label=f"Mean: {mean_acc:.4f}")
    axes[0].fill_between([0.5, FOLDS + 0.5],
                         mean_acc - std_acc, mean_acc + std_acc,
                         alpha=0.15, color="blue", label=f"±1 std: {std_acc:.4f}")
    axes[0].set_xticks(folds)
    axes[0].set_xticklabels([f"Fold {i}" for i in folds], fontsize=10)
    axes[0].set_ylim(max(0, mean_acc - 0.1), 1.02)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_title(f"{FOLDS}-Fold Cross-Validation Accuracy", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(axis="y", alpha=0.3)

    # Right: all metrics mean +- std
    metric_labels = ["Accuracy", "F1 Macro", "Precision", "Recall"]
    metric_keys   = ["test_accuracy", "test_f1_macro", "test_precision_macro", "test_recall_macro"]
    means = [cv_results[k].mean() for k in metric_keys]
    stds  = [cv_results[k].std()  for k in metric_keys]

    bars2 = axes[1].bar(metric_labels, means, edgecolor="white", alpha=0.85)
    axes[1].errorbar(metric_labels, means, yerr=stds, fmt="none",
                     color="black", capsize=6, linewidth=2)
    
    for bar, mean, std in zip(bars2, means, stds):
        axes[1].text(bar.get_x() + bar.get_width() / 2, mean + std + 0.005,
                     f"{mean:.4f}\n±{std:.4f}", ha="center", va="bottom", fontsize=9)
        
    axes[1].set_ylim(max(0, min(means) - 0.1), 1.08)
    axes[1].set_ylabel("Score", fontsize=12)
    axes[1].set_title("CV Metrics - Mean +- Std", fontsize=13, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved to {out}")
    return cv_results


def save_text_report(y_test, y_pred, labels: list, cv_results: dict, out: Path) -> None:
    report = classification_report(y_test, y_pred, labels=labels, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm,
                         index=[f"true_{l}" for l in labels],
                         columns=[f"pred_{l}" for l in labels])
    
    with open(out, "w") as f:
        f.write("GESTURE CLASSIFIER EVALUATION REPORT\n")
        f.write("\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix (counts):\n")
        f.write(cm_df.to_string() + "\n\n")
        f.write(f"{FOLDS}-Fold Cross-Validation Results:\n")
        f.write("\n")

        for k, v in cv_results.items():
            if k.startswith("test_"):
                metric = k.replace("test_", "")
                f.write(f"  {metric:<22} mean={v.mean():.4f}  std={v.std():.4f}\n")

    print(f"Saved to {out}")


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading model and data...")
    artifact = load(args.model)
    model = artifact["model"]
    labels = artifact["class_names"]

    x, y, (x_train, x_test, y_train, y_test) = load_data(Path(args.data))
    y_pred = model.predict(x_test)

    print("\nGenerating plots...")
    plot_confusion_matrix(y_test, y_pred, labels, RESULTS_DIR / "confusion_matrix.png")
    plot_class_metrics(y_test, y_pred, labels, RESULTS_DIR / "class_metrics.png")
    plot_dataset_distribution(y, labels, RESULTS_DIR / "dataset_distribution.png")

    print(f"\nRunning {FOLDS}-fold cross-validation...")
    cv_results = plot_cross_validation(model, x, y, RESULTS_DIR / "cross_validation.png")

    save_text_report(y_test, y_pred, labels, cv_results, RESULTS_DIR / "metrics_report.txt")

    print(f"\nSaved to '{RESULTS_DIR}/'")


if __name__ == "__main__":
    main()