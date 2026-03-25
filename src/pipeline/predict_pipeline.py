import pandas as pd
import numpy as np
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.model_path = "models/model.pkl"

        # BUG FIX 4 applied:
        # Previously, predictions were decoded with a hard-coded:
        #   return ["DDoS" if p == 1 else "BENIGN" for p in preds]
        # This is brittle for two reasons:
        #   1. It assumes DDoS is always encoded as 1 — true for binary alphabetical
        #      encoding, but fragile and undocumented.
        #   2. It completely breaks for multi-class (Option C) where there are many
        #      numeric labels (0,1,2,3...) each mapping to a different attack type.
        #
        # FIX: Load the saved LabelEncoder and use inverse_transform() to decode
        # numeric predictions back to class names. Works for both binary and multi-class.
        self.encoder_path = "models/label_encoder.pkl"

    def predict(self, df: pd.DataFrame):
        model = load_object(self.model_path)
        label_encoder = load_object(self.encoder_path)

        # Strip whitespace from column names (raw inference data may have spaces).
        df = df.copy()
        df.columns = df.columns.str.strip()

        # Drop label column if present (e.g. if running on labelled test data).
        df = df.drop(columns=[col for col in ['Label'] if col in df.columns])

        # Align feature columns to what the model was trained on.
        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
        elif hasattr(model, "feature_names_"):
            expected_cols = list(model.feature_names_)
        else:
            raise Exception(
                "Model does not contain feature names. "
                "Ensure you are using a trained sklearn/XGBoost/CatBoost model."
            )

        # Add any missing columns as 0 (same strategy as DataTransformation).
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        X = df[expected_cols]

        # Handle infinities and missing values.
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        preds_numeric = model.predict(X)

        # BUG FIX 4: Use inverse_transform instead of hard-coded label map.
        # Returns original string labels (e.g. "BENIGN", "DDoS", "DoS Hulk", etc.)
        preds_labels = label_encoder.inverse_transform(preds_numeric)

        return preds_labels.tolist()