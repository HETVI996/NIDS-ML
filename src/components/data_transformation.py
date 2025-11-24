import os 
import pandas as pd 
import sys 
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.exception import CustomException
from src.logger import logger 

class DataTransformation:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        ## Selected imporrtant features 
        self.selected_columns = [
            ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
            'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
            ' Flow Packets/s', 'Flow Bytes/s', ' Fwd Packets/s', ' Bwd Packets/s',
            ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean',
            ' Packet Length Std', ' Packet Length Variance',
            ' Active Mean', ' Active Std', ' Active Max', ' Active Min',
            'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min'
        ]
        
        self.label_column = 'Label'
        
    def transform(self, df):
        try:
            # Remove spaces in column names
            df.columns = df.columns.str.strip()

            # Separate label
            y = df[self.label_column]
            X = df.drop(columns=[self.label_column])

            # Select numeric only
            X = X.select_dtypes(include=['int64', 'float64'])

            # Replace inf values
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X = X.fillna(X.median())
            
            if not hasattr(self, "feature_names"):
                # First time during training
                self.feature_names = X.columns.tolist()
            else:
                # During test transformation
                missing_cols = set(self.feature_names) - set(X.columns)
                for col in missing_cols:
                    X[col] = 0
                X = X[self.feature_names]

            
            # Scale numeric columns
            X = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns
            )

            # Add back label column
            X[self.label_column] = y.values

            return X

   


        except Exception as e:
            raise CustomException(e, sys)

        