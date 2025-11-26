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

        # drop label if exists
        for col in ["Label", " Label"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # model expects exact training columns
        expected_cols = model.feature_names_in_
        X = df[expected_cols]

        # replace inf values
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        # fill missing values
        X.fillna(X.median(), inplace=True)

        preds = model.predict(X)
        return ["DDoS" if p == 1 else "BENIGN" for p in preds]


