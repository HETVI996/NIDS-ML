import os 
import pandas as pd
import sys 
from sklearn.metrics import f1_score, precision_score, recall_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.exception import CustomException
from src.logger import logger 
from src.utils import save_object 

class ModelTrainer:
    def __init__(self):
        self.model_path = "models/model.pkl"
        
    def train(self, train_df, test_df):
        try: 
            logger.info("Starting model training process.")
            
            ## split features and target variable
            X_train = train_df.drop(columns =['Label'])
            X_test = test_df.drop(columns=['Label'])
            y_train = train_df['Label']
            y_test = test_df['Label']

            # Convert BENIGN / DDoS to numeric labels for XGBoost
            label_map = {"BENIGN": 0, "DDoS": 1}
            y_train = y_train.map(label_map)
            y_test = y_test.map(label_map)

            
            ## Define Models 
            models = {
                "Random Forest": RandomForestClassifier(n_estimators=50, n_jobs=-1),
                "XGBoost": XGBClassifier(learning_rate=0.1, n_estimators=200, max_depth=8, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss"),
                "CatBoost": CatBoostClassifier(verbose=False),
                "VotingEnsemble": VotingClassifier(estimators=[
                    ('rf', RandomForestClassifier(class_weight="balanced", random_state=42 )),
                    ("xgb", XGBClassifier(
                        learning_rate=0.1,
                        max_depth=8,
                        n_estimators=200,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss"
                    )),
                ],
                voting="soft",
                weights=[1, 2]
                )
            }
    
            best_model = None
            best_score = 0
    
            # Train and evaluate each model
            for name, model in models.items():
                logger.info(f"Training {name} model.. ")
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                f1 = f1_score(y_test, y_pred, pos_label=1)
                precision = precision_score(y_test, y_pred, pos_label=1)
                recall = recall_score(y_test, y_pred, pos_label=1)

                
                logger.info(f"{name} -> F1: {f1}, Precision: {precision}, Recall: {recall}")
                
                if f1 > best_score:
                    best_score = f1
                    best_model = model
                
            # Save the best model
            os.makedirs("models", exist_ok=True)
            save_object(self.model_path, best_model)
            logger.info("Best model saved successfully.")
                
            return best_model, best_score
    
        except Exception as e:
            raise CustomException(e, sys)