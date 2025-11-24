import pandas as pd 
import numpy as np 
from src.utils import load_object 

class PredictPipeline:
    def __init__(self):
        self.model_path = "models/model.pkl"
        
    def predict(self, df):
        model = load_object(self.model_path)

        # fix column names
        df.columns = df.columns.str.strip()

        # numeric features only
        X = df.select_dtypes(include=['int64', 'float64'])

        # handle infinite values
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        # fill missing values
        X.fillna(X.median(), inplace=True)

        # match training feature order
        feature_names = model.feature_names_in_
        missing_cols = set(feature_names) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        X = X[feature_names]  # reorder

        # predict
        predictions = model.predict(X)
        return predictions
