import pandas as pd 
import numpy as np 
from src.utils import load_object 

class PredictPipeline:
    def __init__(self):
        self.model_path = "models/model.pkl"
        
    def predict(self, df: pd.DataFrame):

            model = load_object(self.model_path)

            # clean column names
            df.columns = df.columns.str.strip()

            # drop label column if present
            for col in ["Label", " Label"]:
                if col in df.columns:
                    df = df.drop(columns=[col])

            # ---------------------------------------------
            # FIX: Handle models with different feature name formats
            # ---------------------------------------------
            if hasattr(model, "feature_names_in_"):
                expected_cols = list(model.feature_names_in_)
            elif hasattr(model, "feature_names_"):
                expected_cols = list(model.feature_names_)
            else:
                raise Exception("Model does not contain feature names attribute.")

            # select only the features used in training
            X = df[expected_cols]

            # replace infinities
            X.replace([np.inf, -np.inf], np.nan, inplace=True)

            # fill missing values
            X.fillna(X.median(), inplace=True)

            preds = model.predict(X)

            # convert numeric predictions back to labels
            return ["DDoS" if p == 1 else "BENIGN" for p in preds]


