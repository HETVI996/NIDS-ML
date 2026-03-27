import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


class ModelTrainer:
    def __init__(self):
        self.model_path = "models/model.pkl"
        self.encoder_path = "models/label_encoder.pkl"  # BUG FIX 4: persist encoder

        # BUG FIX 4 (Label encoding not persisted):
        # Previously, the label_map dict {"BENIGN": 0, "DDoS": 1} was defined as a
        # local variable inside train(). This means:
        #   a) It only works for binary classification — breaks for multi-class (Option C).
        #   b) The mapping was never saved, so predict_pipeline.py had to hard-code
        #      "DDoS if p==1 else BENIGN" — which breaks completely for multi-class.
        #
        # FIX: Use sklearn's LabelEncoder, stored on self and saved to disk.
        # predict_pipeline.py can then load and use it for inverse_transform(),
        # converting numeric predictions back to original class names regardless
        # of how many classes there are.
        self.label_encoder = LabelEncoder()

    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        try:
            logger.info("Starting model training process.")

            # Split features and labels.
            X_train = train_df.drop(columns=['Label'])
            X_test = test_df.drop(columns=['Label'])
            y_train_raw = train_df['Label']
            y_test_raw = test_df['Label']

            # BUG FIX 4 applied: Encode labels with LabelEncoder.
            # fit() on train labels, transform() on both — this mirrors the
            # same train/test discipline applied to the scaler in DataTransformation.
            # For binary: encodes BENIGN->0, DDoS->1 (alphabetical order).
            # For multi-class (Option C): encodes all attack types automatically.
            y_train = self.label_encoder.fit_transform(y_train_raw)
            y_test = self.label_encoder.transform(y_test_raw)

            n_classes = len(self.label_encoder.classes_)
            logger.info(f"Classes detected ({n_classes}): {list(self.label_encoder.classes_)}")

            # Determine averaging strategy for metrics:
            # - Binary: average='binary' with pos_label for the attack class.
            # - Multi-class: average='weighted' to account for class imbalance.
            is_binary = (n_classes == 2)
            if is_binary:
                # pos_label is the integer encoding of "DDoS" (or whichever attack class).
                # We find it dynamically rather than hard-coding 1.
                attack_classes = [c for c in self.label_encoder.classes_ if c != 'BENIGN']
                pos_label = int(self.label_encoder.transform([attack_classes[0]])[0])
                avg_strategy = 'binary'
                logger.info(f"Binary mode. Attack class: '{attack_classes[0]}', pos_label={pos_label}")
            else:
                pos_label = None  # not used for multi-class
                avg_strategy = 'weighted'
                logger.info("Multi-class mode. Using weighted averaging for metrics.")

            # Define models.
            # NOTE: XGBClassifier requires num_class for multi-class with softmax.
            xgb_params = dict(
                learning_rate=0.1,
                n_estimators=200,
                max_depth=8,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="mlogloss" if not is_binary else "logloss",
            )
            if not is_binary:
                xgb_params["objective"] = "multi:softprob"
                xgb_params["num_class"] = n_classes

            models = {
                "RandomForest": RandomForestClassifier(
                    n_estimators=50,
                    n_jobs=-1,
                    class_weight="balanced",  # handles class imbalance
                    random_state=42
                ),
                "XGBoost": XGBClassifier(**xgb_params),
                "CatBoost": CatBoostClassifier(verbose=False, random_state=42),
                "VotingEnsemble": VotingClassifier(
                    estimators=[
                        ('rf', RandomForestClassifier(
                            class_weight="balanced", random_state=42
                        )),
                        ('xgb', XGBClassifier(**xgb_params)),
                    ],
                    voting="soft",
                    weights=[1, 2]
                ),
            }

            best_model = None
            best_score = 0.0

            for name, model in models.items():
                logger.info(f"Training {name}...")
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                if is_binary:
                    f1 = f1_score(y_test, y_pred, average=avg_strategy, pos_label=pos_label)
                    precision = precision_score(y_test, y_pred, average=avg_strategy, pos_label=pos_label)
                    recall = recall_score(y_test, y_pred, average=avg_strategy, pos_label=pos_label)
                else:
                    f1 = f1_score(y_test, y_pred, average=avg_strategy)
                    precision = precision_score(y_test, y_pred, average=avg_strategy)
                    recall = recall_score(y_test, y_pred, average=avg_strategy)

                logger.info(
                    f"{name} -> F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
                )

                # Full per-class report — essential for publication (reviewers expect this).
                report = classification_report(
                    y_test, y_pred,
                    target_names=self.label_encoder.classes_
                )
                logger.info(f"\n{name} Classification Report:\n{report}")

                if f1 > best_score:
                    best_score = f1
                    best_model = model
                    best_name = name

            logger.info(f"Best model: {best_name} with F1={best_score:.4f}")

            # Save the best model.
            os.makedirs("models", exist_ok=True)
            save_object(self.model_path, best_model)

            # BUG FIX 4 applied: Save the label encoder so predict_pipeline.py
            # can decode numeric predictions back to class names without hard-coding.
            save_object(self.encoder_path, self.label_encoder)

            logger.info("Best model and label encoder saved successfully.")

            return best_model, best_score

        except Exception as e:
            raise CustomException(e, sys)