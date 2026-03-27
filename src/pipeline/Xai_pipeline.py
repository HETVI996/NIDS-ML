"""
xai_pipeline.py

Explainability analysis for the trained DDoS detection model.
Produces two categories of explanations required for publication:

  1. GLOBAL explanations (SHAP):
     - Feature importance bar chart (which features matter most overall)
     - Beeswarm plot per attack class (how each feature pushes prediction
       toward or away from a specific class)
     This answers the paper question: "what does the model rely on?"

  2. LOCAL explanations (LIME):
     - Per-instance explanation for a sample of misclassified flows
     This answers the paper question: "why did the model fail on these cases?"

Both sets of outputs are saved to reports/ as PNG files ready for the paper.

Usage:
    python xai_pipeline.py

Requirements (add to requirements.txt):
    shap>=0.44
    lime>=0.2.0.1
    matplotlib>=3.7
    seaborn>=0.13
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
import seaborn as sns

import shap
from lime.lime_tabular import LimeTabularExplainer

from src.utils import load_object
from src.components.data_transformation import DataTransformation
from src.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH        = "models/model.pkl"
ENCODER_PATH      = "models/label_encoder.pkl"
TEST_DATA_PATH    = "data/processed/test.csv"
TRAIN_DATA_PATH   = "data/processed/train.csv"
REPORTS_DIR       = "reports"

# How many test samples to run SHAP on.
# SHAP on a full test set is slow (especially for VotingEnsemble).
# 2000 is enough for stable global importance plots; use more if you have time.
SHAP_SAMPLE_SIZE  = 2000

# Number of misclassified flows to explain with LIME.
LIME_N_SAMPLES    = 10

os.makedirs(REPORTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load model, encoder, and test data
# ─────────────────────────────────────────────────────────────────────────────

def load_artifacts():
    logger.info("Loading model and label encoder...")
    model = load_object(MODEL_PATH)
    label_encoder = load_object(ENCODER_PATH)
    return model, label_encoder


def load_test_data(label_encoder):
    """
    Loads the processed test CSV and returns X_test (features only) and y_test
    (numeric labels as encoded during training).

    We re-apply DataTransformation.transform() here so that the scaler is
    consistent with what was used during training.

    IMPORTANT: We need the transformer that was fit on training data.
    Since we saved the model but not the transformer, we refit on train data
    here. This is the correct approach — the transformer is refit on training
    data, NOT on test data.
    """
    logger.info("Loading and transforming test data...")

    train_df_raw = pd.read_csv(TRAIN_DATA_PATH)
    test_df_raw  = pd.read_csv(TEST_DATA_PATH)

    transformer = DataTransformation()
    train_df = transformer.fit_transform(train_df_raw)  # fit on train
    test_df  = transformer.transform(test_df_raw)       # transform test only

    X_test    = test_df.drop(columns=["Label"])
    y_test_str = test_df["Label"]

    # Encode labels to numeric using the saved encoder.
    # We use transform() here (not fit_transform) to apply the exact same
    # mapping that was used during model training.
    y_test = label_encoder.transform(y_test_str)

    feature_names = X_test.columns.tolist()

    return X_test.values, y_test, feature_names, label_encoder.classes_


# ─────────────────────────────────────────────────────────────────────────────
# 2. Resolve the underlying tree model from VotingEnsemble if needed
# ─────────────────────────────────────────────────────────────────────────────

def get_shap_compatible_model(model):
    """
    SHAP's TreeExplainer works on tree-based models (RF, XGB, CatBoost).
    It does NOT directly support VotingClassifier (which is an ensemble wrapper).

    Strategy:
    - If the saved model IS a VotingClassifier, extract its XGBoost sub-estimator
      for SHAP. XGBoost produces the most interpretable SHAP values and is the
      model with the highest weight in the ensemble (weight=2).
    - If the model is already RF/XGB/CatBoost, use it directly.

    This is documented in the paper as: "SHAP values were computed on the
    XGBoost component of the ensemble, which carries the highest weight."
    """
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier
    from xgboost import XGBClassifier

    if isinstance(model, VotingClassifier):
        logger.info(
            "Model is VotingClassifier. Extracting XGBoost sub-estimator for SHAP."
        )
        for name, estimator in model.estimators_:
            if isinstance(estimator, XGBClassifier):
                logger.info(f"Using sub-estimator: {name}")
                return estimator
        # Fallback: use the first estimator if no XGB found
        logger.warning("No XGBClassifier found in VotingClassifier. Using first estimator.")
        return model.estimators_[0][1]

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. SHAP: Global feature importance
# ─────────────────────────────────────────────────────────────────────────────

def run_shap_analysis(model, X_test, feature_names, class_names):
    """
    Runs SHAP TreeExplainer on a sample of the test set.

    Produces:
      - reports/shap_global_importance.png  — bar chart of mean |SHAP| per feature
      - reports/shap_beeswarm_{class}.png   — beeswarm for each attack class

    Why SHAP TreeExplainer specifically:
    - Exact (not approximate) for tree-based models
    - Produces per-class SHAP values for multi-class models
    - Much faster than KernelExplainer for tree ensembles
    """
    logger.info(f"Running SHAP on {SHAP_SAMPLE_SIZE} test samples...")

    shap_model = get_shap_compatible_model(model)

    # Sample for speed — stratified if possible
    n = min(SHAP_SAMPLE_SIZE, len(X_test))
    idx = np.random.choice(len(X_test), size=n, replace=False)
    X_sample = X_test[idx]

    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X_sample)
    # shap_values is a list of arrays if multi-class, or a single array if binary.
    # Shape per class: (n_samples, n_features)

    is_multiclass = isinstance(shap_values, list)

    # ── 3a. Global feature importance bar chart ──────────────────────────────
    # Mean absolute SHAP value across all classes and samples.
    # This is the standard way to rank features in a multi-class setting.
    if is_multiclass:
        mean_abs_shap = np.mean(
            [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
        )
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df.head(20),
        x="mean_abs_shap",
        y="feature",
        palette="viridis"
    )
    plt.title("Top 20 features by mean |SHAP| value (all classes)", fontsize=13)
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.tight_layout()
    out_path = os.path.join(REPORTS_DIR, "shap_global_importance.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {out_path}")

    # ── 3b. Beeswarm plot per class ──────────────────────────────────────────
    # For each attack class, shows how feature values (colour) push the
    # prediction up or down. This is the key figure for the paper — it shows
    # which flow statistics distinguish each attack type from others.
    if is_multiclass:
        for class_idx, class_name in enumerate(class_names):
            safe_name = class_name.replace(" ", "_").replace("/", "-")
            fig, ax = plt.subplots(figsize=(10, 7))
            shap.summary_plot(
                shap_values[class_idx],
                X_sample,
                feature_names=feature_names,
                show=False,
                plot_type="violin"
            )
            plt.title(f"SHAP summary — class: {class_name}", fontsize=12)
            plt.tight_layout()
            out_path = os.path.join(REPORTS_DIR, f"shap_beeswarm_{safe_name}.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved: {out_path}")
    else:
        # Binary case: single beeswarm
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title("SHAP summary — binary classification", fontsize=12)
        plt.tight_layout()
        out_path = os.path.join(REPORTS_DIR, "shap_beeswarm.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {out_path}")

    return shap_values, idx


# ─────────────────────────────────────────────────────────────────────────────
# 4. LIME: Local explanations on misclassified flows
# ─────────────────────────────────────────────────────────────────────────────

def run_lime_analysis(model, X_test, y_test, feature_names, class_names):
    """
    Finds misclassified flows in the test set and generates a LIME explanation
    for each of the first LIME_N_SAMPLES misclassifications.

    Why LIME on misclassified flows:
    - These are the paper's "error analysis" section
    - They reveal which feature patterns confused the model
    - Reviewers expect explanation of failure modes, not just successes

    Output:
      - reports/lime_misclassified_{i}.png for i in range(LIME_N_SAMPLES)
    """
    logger.info("Finding misclassified test samples for LIME...")

    y_pred = model.predict(X_test)
    misclassified_idx = np.where(y_pred != y_test)[0]

    if len(misclassified_idx) == 0:
        logger.info("No misclassifications found. Skipping LIME.")
        return

    logger.info(
        f"Found {len(misclassified_idx)} misclassified samples. "
        f"Explaining first {LIME_N_SAMPLES}."
    )

    # LimeTabularExplainer requires the training data statistics (or a sample)
    # to understand the feature distributions. X_test is acceptable here as a
    # proxy — for publication, use X_train for more accurate neighbourhood sampling.
    explainer = LimeTabularExplainer(
        training_data=X_test,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )

    for i, idx in enumerate(misclassified_idx[:LIME_N_SAMPLES]):
        instance = X_test[idx]
        true_label = class_names[y_test[idx]]
        pred_label = class_names[y_pred[idx]]

        explanation = explainer.explain_instance(
            data_row=instance,
            predict_fn=model.predict_proba,
            num_features=10,
            top_labels=1
        )

        fig = explanation.as_pyplot_figure(label=y_pred[idx])
        fig.suptitle(
            f"LIME — sample {idx}  |  True: {true_label}  |  Predicted: {pred_label}",
            fontsize=10
        )
        plt.tight_layout()
        out_path = os.path.join(REPORTS_DIR, f"lime_misclassified_{i}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, label_encoder = load_artifacts()
    X_test, y_test, feature_names, class_names = load_test_data(label_encoder)

    run_shap_analysis(model, X_test, feature_names, class_names)
    run_lime_analysis(model, X_test, y_test, feature_names, class_names)

    logger.info(f"XAI analysis complete. All outputs saved to: {REPORTS_DIR}/")
    print(f"Done. Check {REPORTS_DIR}/ for SHAP and LIME plots.")