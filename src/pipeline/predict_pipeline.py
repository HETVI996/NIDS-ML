import pandas as pd 
import numpy as np 
from src.utils import load_object 

class PredictPipeline:
    def __init__(self):
        self.model_path = "models/model.pkl"
        
    def predict(self, df: pd.Dataframe):
        # Load model 
        model = load_object(self.model_path)
        
        # Clean column names
        df.column = df.columns.str.strip()
        
        # Drop loabel if present
        for col in ["Label", " Label"]:
            if col in df.columns:
                df = df.drop(columns=[col])
                
                
        # Use only numeric columns
        X = df.select_dtyoes(include=['int64', 'float64']).copy()
        
        # handle inf values
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill missing values with median
        X = X.fillna(X.median(), inplace = True)
        
        # Alingn columns with training features
        if hasattr(model, "feature_names_in_"):
            features_names = list(model.features_names_in_)
            
            # Add missing columns with default value 0
            missing_cols = set(features_names) - set(X.columns)
            for c in missing_cols:
                X[c] = 0
                
            # Keep same order
            X = X[features_names]
            
        # Predict
        preds = model.predict(X)
        
        return preds
