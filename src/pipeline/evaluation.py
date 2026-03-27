"""
evaluation.py

Generates all evaluation figures and tables required for a publishable paper:

  1. Classification report (precision, recall, F1 per class) — saved as CSV + PNG table
  2. Confusion matrix (normalised) — saved as PNG heatmap
  3. ROC curves per class (one-vs-rest) with AUC scores — saved as PNG

These three outputs directly map to the Results section tables and figures
that every IDS/network security paper is expected to contain.

Usage:
    python evaluation.py

Requirements (add to requirements.txt):
    matplotlib>=3.7
    seaborn>=0.13
    scikit-learn>=1.3
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from src.utils import load_object
from src.components.data_transformation import DataTransformation
from src.logger import logger


MODEL_PATH     = "models/model.pkl"
ENCODER_PATH   = "models/label_encoder.pkl"
TRAIN_PATH     = "data/processed/train.csv"
TEST_PATH      = "data/processed/test.csv"
REPORTS_DIR    = "reports"

os.makedirs(REPORTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load and prepare data
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data():
    model         = load_object(MODEL_PATH)
    label_encoder = load_object(ENCODER_PATH)

    train_df_raw  = pd.read_csv(TRAIN_PATH)
    test_df_raw   = pd.read_csv(TEST_PATH)

    transformer   = DataTransformation()
    train_df      = transformer.fit_transform(train_df_raw)
    test_df       = transformer.transform(test_df_raw)

    X_test        = test_df.drop(columns=["Label"]).values
    y_test_str    = test_df["Label"]
    y_test        = label_encoder.transform(y_test_str)
    y_pred        = model.predict(X_test)

    # Predicted probabilities needed for ROC — not all models support predict_proba.
    # VotingClassifier with voting='soft' does. Others generally do.
    try:
        y_prob = model.predict_proba(X_test)
    except AttributeError:
        logger.warning("Model does not support predict_proba. ROC curves will be skipped.")
        y_prob = None

    return model, label_encoder, X_test, y_test, y_pred, y_prob


# ─────────────────────────────────────────────────────────────────────────────
# 1. Classification report → CSV + styled PNG table
# ─────────────────────────────────────────────────────────────────────────────

def save_classification_report(y_test, y_pred, class_names):
    """
    Saves:
      - reports/classification_report.csv   (for paper table)
      - reports/classification_report.png   (for paper figure / supplementary)
    """
    report_dict = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True
    )

    # Convert to DataFrame, drop 'support' (it's raw count, less useful in a plot)
    df_report = pd.DataFrame(report_dict).T
    df_report = df_report[["precision", "recall", "f1-score", "support"]]
    df_report = df_report.round(4)

    csv_path = os.path.join(REPORTS_DIR, "classification_report.csv")
    df_report.to_csv(csv_path)
    logger.info(f"Saved: {csv_path}")

    # Styled table as PNG — shows per-class metrics clearly
    fig, ax = plt.subplots(figsize=(10, len(class_names) * 0.5 + 3))
    ax.axis("off")

    table = ax.table(
        cellText=df_report.round(4).values,
        rowLabels=df_report.index,
        colLabels=df_report.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)

    # Colour-code F1 column: higher = greener
    f1_col_idx = df_report.columns.tolist().index("f1-score")
    for row_idx in range(len(df_report)):
        val = df_report.iloc[row_idx]["f1-score"]
        if pd.notna(val) and isinstance(val, float):
            alpha = min(max(val, 0), 1) * 0.4
            table[row_idx + 1, f1_col_idx].set_facecolor((0.2, 0.8, 0.4, alpha))

    plt.title("Classification report — per class", fontsize=12, pad=12)
    plt.tight_layout()
    png_path = os.path.join(REPORTS_DIR, "classification_report.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {png_path}")

    # Also print to console for quick inspection
    print("\n" + classification_report(y_test, y_pred, target_names=class_names))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Normalised confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def save_confusion_matrix(y_test, y_pred, class_names):
    """
    Saves reports/confusion_matrix.png

    Normalised by true label (row-normalised) so that class imbalance doesn't
    make small attack classes invisible.  A perfectly normalised matrix has 1.0
    on every diagonal cell.

    For a paper: include this as Figure X in the Results section.
    """
    cm = confusion_matrix(y_test, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.7)))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.3,
        ax=ax
    )
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title("Normalised confusion matrix", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    out_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ROC curves per class (one-vs-rest)
# ─────────────────────────────────────────────────────────────────────────────

def save_roc_curves(y_test, y_prob, class_names):
    """
    Saves reports/roc_curves.png

    One-vs-rest ROC: for each class, the model's ability to distinguish
    "this class vs everything else". AUC values are annotated on each curve.

    For publication: AUC scores are commonly reported in Table 2 and the
    ROC figure goes in supplementary material (or main if space allows).
    """
    if y_prob is None:
        logger.warning("Skipping ROC curves — no predict_proba available.")
        return

    n_classes = len(class_names)

    # Binarize y_test for one-vs-rest
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

    auc_scores = {}
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[class_name] = round(roc_auc, 4)
        ax.plot(
            fpr, tpr,
            color=color,
            lw=1.5,
            label=f"{class_name} (AUC={roc_auc:.3f})"
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.set_title("ROC curves — one-vs-rest per class", fontsize=12)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    plt.tight_layout()

    out_path = os.path.join(REPORTS_DIR, "roc_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")

    # Save AUC scores as CSV for the paper table
    auc_df = pd.DataFrame.from_dict(auc_scores, orient="index", columns=["ROC-AUC"])
    auc_csv = os.path.join(REPORTS_DIR, "auc_scores.csv")
    auc_df.to_csv(auc_csv)
    logger.info(f"Saved: {auc_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, label_encoder, X_test, y_test, y_pred, y_prob = prepare_data()
    class_names = label_encoder.classes_

    save_classification_report(y_test, y_pred, class_names)
    save_confusion_matrix(y_test, y_pred, class_names)
    save_roc_curves(y_test, y_prob, class_names)

    logger.info(f"Evaluation complete. All outputs in: {REPORTS_DIR}/")
    print(f"\nAll evaluation figures saved to {REPORTS_DIR}/")
    print("Files produced:")
    for f in sorted(os.listdir(REPORTS_DIR)):
        print(f"  {REPORTS_DIR}/{f}")