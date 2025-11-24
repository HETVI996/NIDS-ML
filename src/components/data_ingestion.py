import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logger

class DataIngestion:
    def __init__(self):
        self.raw_data_path = "data/raw/Friday_DDos.csv"
        self.train_data_path = "data/processed/train.csv"
        self.test_data_path = "data/processed/test.csv"

    def initiate_data_ingestion(self):
        logger.info("Data Ingestion started.")

        try:
            # if data already exists skip ingestion
            if os.path.exists(self.train_data_path) and os.path.exists(self.test_data_path):
                logger.info("Train and Test data already exists. Skipping ingestion.")
                return self.train_data_path, self.test_data_path

            df = pd.read_csv(self.raw_data_path, low_memory=False)
            logger.info("Dataset loaded successfully.")

            processed_dir = os.path.dirname(self.train_data_path)
            os.makedirs(processed_dir, exist_ok=True)

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df[' Label']
            )

            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)

            logger.info("Train and Test files saved successfully.")
            return self.train_data_path, self.test_data_path

        except Exception as e:
            raise CustomException(e, sys)

        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()
