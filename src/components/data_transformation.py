import os
import pandas as pd
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.exception import CustomException
from src.logger import logger


class DataTransformation:
    def __init__(self):
        # BUG FIX 1 (Scaler refit):
        # Previously, a new DataTransformation() instance was created for both
        # train and test in train_pipeline.py, meaning each got its own fresh
        # StandardScaler. Calling fit_transform() on each independently caused
        # the test set to be scaled with its own mean/std — not the training set's.
        # This breaks the fundamental rule: test data must be transformed using
        # statistics learned ONLY from training data.
        #
        # FIX: A single DataTransformation instance must now be used for both
        # train and test. fit_transform() is called only for train; transform()
        # only for test. The scaler is stored on self so state persists.
        self.scaler = StandardScaler()
        self._scaler_fitted = False  # tracks whether fit has been called yet

        # BUG FIX 2 (selected_columns never applied):
        # Previously, self.selected_columns was defined here but never used in
        # transform(). The code did X = X.select_dtypes(include=['int64','float64'])
        # which selected ALL numeric columns — ignoring this list entirely.
        #
        # FIX: transform() now explicitly filters to self.selected_columns.
        # These 22 features are the domain-relevant flow statistics for DDoS detection.
        # Any column in this list missing from the data gets filled with 0 (handled below).
        self.selected_columns = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Flow Packets/s', 'Flow Bytes/s', 'Fwd Packets/s', 'Bwd Packets/s',
            'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
            'Packet Length Std', 'Packet Length Variance',
            'Active Mean', 'Active Std', 'Active Max', 'Active Min',
            'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
        ]
        # NOTE: Column names are stripped of leading/trailing spaces in transform()
        # before this list is matched, so no leading-space variants are needed here.

        self.label_column = 'Label'

        # BUG FIX 3 (feature_names logic):
        # Previously, feature_names was set inside transform() using hasattr(self, 'feature_names').
        # But since self.feature_names was never set in __init__, the hasattr check
        # would always be False on the first call (train) — so far so good — but on the
        # second call (test), it would enter the else branch. HOWEVER, since a brand-new
        # DataTransformation() was created for test in train_pipeline.py, self.feature_names
        # was never set on it either, so it would AGAIN take the "first time" path,
        # re-learning feature names from test data. This combined with Bug Fix 1 above.
        #
        # FIX: feature_names is initialized to None here in __init__ so the check
        # in transform() is explicit and unambiguous.
        self.feature_names = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Call this ONLY on training data.
        Learns the scaler statistics and feature column order from training data,
        then returns the scaled DataFrame with label column preserved.
        """
        return self._transform_internal(df, is_train=True)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Call this ONLY on test/inference data, AFTER fit_transform() has been called.
        Applies the already-fitted scaler — no re-fitting.
        Raises if called before fit_transform().
        """
        if not self._scaler_fitted:
            raise RuntimeError(
                "DataTransformation.transform() called before fit_transform(). "
                "Always call fit_transform(train_df) first."
            )
        return self._transform_internal(df, is_train=False)

    def _transform_internal(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        try:
            # Step 1: Strip leading/trailing whitespace from all column names.
            # CIC-IDS2017 has many columns like ' Label', ' Flow Duration', etc.
            # Stripping here means self.selected_columns (no spaces) will match correctly.
            df = df.copy()
            df.columns = df.columns.str.strip()

            # Step 2: Separate label from features.
            if self.label_column not in df.columns:
                raise ValueError(
                    f"Label column '{self.label_column}' not found. "
                    f"Available columns: {df.columns.tolist()}"
                )
            y = df[self.label_column].values
            X = df.drop(columns=[self.label_column])

            # Step 3: Select only numeric columns first (drops any stray string columns).
            X = X.select_dtypes(include=['int64', 'float64'])

            # Step 4 (BUG FIX 2 applied): Filter to the selected feature columns.
            # For any selected column missing in this particular file, add it as 0.
            # This handles multi-file datasets (Option C) where different day-files
            # may have slightly different column sets.
            for col in self.selected_columns:
                if col not in X.columns:
                    logger.warning(f"Selected column '{col}' not found in data. Filling with 0.")
                    X[col] = 0
            X = X[self.selected_columns]

            # Step 5: Replace infinities with NaN, then fill NaN with column medians.
            # Infinity values arise from rate features (e.g. Flow Bytes/s = bytes / 0-duration flows).
            X = X.replace([np.inf, -np.inf], np.nan)
            if is_train:
                # Compute and store medians from training data only.
                self._fill_medians = X.median()
            X = X.fillna(self._fill_medians)

            # Step 6 (BUG FIX 1 + BUG FIX 3 applied):
            # fit_transform only on train; transform-only on test.
            if is_train:
                X_scaled = self.scaler.fit_transform(X)
                self._scaler_fitted = True
                self.feature_names = X.columns.tolist()  # BUG FIX 3: stored explicitly
                logger.info(f"Scaler fitted on training data. Features: {self.feature_names}")
            else:
                # Reorder columns to match training feature order (safety check).
                X = X[self.feature_names]
                X_scaled = self.scaler.transform(X)  # Uses training mean/std — correct.

            X_out = pd.DataFrame(X_scaled, columns=self.feature_names)

            # Step 7: Re-attach label column.
            X_out[self.label_column] = y

            return X_out

        except Exception as e:
            raise CustomException(e, sys)