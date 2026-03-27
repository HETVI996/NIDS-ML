"""
Xai_pipeline.py — SHAP + LIME explainability for the trained NIDS model.

Outputs saved to reports/:
  - shap_global_importance.png
  - shap_beeswarm_{class}.png  (one per class)
  - lime_misclassified_{i}.png (up to LIME_N_SAMPLES)

Run from project root:
    python -m src.pipeline.Xai_pipeline
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import shap
from lime.lime_tabular import LimeTabularExplainer

from src.utils import load_object
from src.components.data_transformation import DataTransformation
from src.logger import logger

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH       = "models/model.pkl"
ENCODER_PATH     = "models/label_encoder.pkl"
TRAIN_DATA_PATH  = "data/processed/train.csv"
TEST_DATA_PATH   = "data/processed/test.csv"
REPORTS_DIR      = "reports"
SHAP_SAMPLE_SIZE = 2000
LIME_N_SAMPLES   = 10

os.makedirs(REPORTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load artifacts
# ─────────────────────────────────────────────────────────────────────────────
def load_artifacts():
    logger.info("Loading model and label encoder...")
    model         = load_object(MODEL_PATH)
    label_encoder = load_object(ENCODER_PATH)
    return model, label_encoder


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load and transform test data
# ─────────────────────────────────────────────────────────────────────────────
def load_test_data(label_encoder):
    logger.info("Loading and transforming data...")

    train_df_raw = pd.read_csv(TRAIN_DATA_PATH, low_memory=False)
    test_df_raw  = pd.read_csv(TEST_DATA_PATH,  low_memory=False)

    transformer  = DataTransformation()
    train_df     = transformer.fit_transform(train_df_raw)
    test_df      = transformer.transform(test_df_raw)

    X_test       = test_df.drop(columns=["Label"])
    y_test_str   = test_df["Label"]
    y_test       = label_encoder.transform(y_test_str)
    feature_names = X_test.columns.tolist()

    # Return as DataFrame to avoid feature-name warnings from sklearn
    return X_test, y_test, feature_names, label_encoder.classes_


# ─────────────────────────────────────────────────────────────────────────────
# 3. Extract SHAP-compatible single tree model from VotingClassifier
# ─────────────────────────────────────────────────────────────────────────────
def get_shap_compatible_model(model):
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier

    if isinstance(model, VotingClassifier):
        logger.info("Model is VotingClassifier. Extracting RandomForest sub-estimator for SHAP.")
        for estimator in model.estimators_:
            if isinstance(estimator, RandomForestClassifier):
                logger.info("Using RandomForest sub-estimator for SHAP.")
                return estimator
        logger.warning("No RandomForestClassifier found. Using first sub-estimator.")
        return model.estimators_[0]

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. SHAP global importance + per-class beeswarm
# ─────────────────────────────────────────────────────────────────────────────
def run_shap_analysis(model, X_test, feature_names, class_names):
    logger.info(f"Running SHAP on up to {SHAP_SAMPLE_SIZE} test samples...")

    shap_model = get_shap_compatible_model(model)

    n   = min(SHAP_SAMPLE_SIZE, len(X_test))
    idx = np.random.choice(len(X_test), size=n, replace=False)
    X_sample = X_test.iloc[idx].values

    explainer   = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X_sample)


    # Normalise to consistent format: list of 2D arrays, one per class.
    # RandomForest returns 3D array (n_classes, n_samples, n_features).
    # XGBoost binary returns 2D array (n_samples, n_features).
    # XGBoost multi returns list of 2D arrays.
    # Shape is (n_samples, n_features, n_classes) — this SHAP version's RF format.
    # Reorder to (n_classes, n_samples, n_features) for uniform handling.
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # (2000, 22, 2) → list of two (2000, 22) arrays, one per class
        shap_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    elif isinstance(shap_values, list):
        shap_list = shap_values
    else:
        shap_list = [shap_values]

    is_multiclass = len(shap_list) > 1

    # Global importance: mean |SHAP| across all classes → shape (n_features,)
    mean_abs_shap = np.mean(
        [np.abs(sv).mean(axis=0) for sv in shap_list], axis=0
    ).flatten()  # ensure 1D

    importance_df = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df.head(20),
        x="mean_abs_shap",
        y="feature",
        palette="viridis"
    )
    plt.title("Top 20 features by mean |SHAP| value", fontsize=13)
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, "shap_global_importance.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Saved: {out}")

    # Beeswarm per class
    for class_idx, class_name in enumerate(class_names):
        safe = class_name.replace(" ", "_").replace("/", "-")
        shap.summary_plot(
            shap_list[class_idx], X_sample,
            feature_names=feature_names,
            show=False, plot_type="violin"
        )
        plt.title(f"SHAP summary — {class_name}", fontsize=12)
        plt.tight_layout()
        out = os.path.join(REPORTS_DIR, f"shap_beeswarm_{safe}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {out}")
# ─────────────────────────────────────────────────────────────────────────────
# 5. LIME on misclassified flows
# ─────────────────────────────────────────────────────────────────────────────
def run_lime_analysis(model, X_test, y_test, feature_names, class_names):
    logger.info("Finding misclassified samples for LIME...")

    X_arr  = X_test.values
    y_pred = model.predict(X_arr)
    misclassified_idx = np.where(y_pred != y_test)[0]

    if len(misclassified_idx) == 0:
        logger.info("No misclassifications found — perfect predictions. Skipping LIME.")
        print("No misclassified samples found. LIME skipped.")
        return

    logger.info(
        f"{len(misclassified_idx)} misclassified samples found. "
        f"Explaining first {LIME_N_SAMPLES}."
    )

    explainer = LimeTabularExplainer(
        training_data=X_arr,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )

    for i, idx in enumerate(misclassified_idx[:LIME_N_SAMPLES]):
        true_label = class_names[y_test[idx]]
        pred_label = class_names[y_pred[idx]]

        exp = explainer.explain_instance(
            data_row=X_arr[idx],
            predict_fn=model.predict_proba,
            num_features=10,
            top_labels=1
        )
        fig = exp.as_pyplot_figure(label=int(y_pred[idx]))
        fig.suptitle(
            f"LIME — sample {idx}  |  True: {true_label}  |  Predicted: {pred_label}",
            fontsize=10
        )
        plt.tight_layout()
        out = os.path.join(REPORTS_DIR, f"lime_misclassified_{i}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, label_encoder    = load_artifacts()
    X_test, y_test, feature_names, class_names = load_test_data(label_encoder)

    run_shap_analysis(model, X_test, feature_names, class_names)
    run_lime_analysis(model, X_test, y_test, feature_names, class_names)

    logger.info(f"XAI complete. Outputs in: {REPORTS_DIR}/")
    print(f"\nDone. Check {REPORTS_DIR}/ for SHAP and LIME plots.")