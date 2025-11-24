import pandas as pd 
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logger

if __name__ == "__main__":

        logger.info("Starting training pipeline...")
        
        ## Load Train and Test csv
        
        train_path, test_path = DataIngestion().initiate_data_ingestion()
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        ## Apply data transformation
        transformer = DataTransformation()
        train_df = transformer.transform(train_df)
        test_df = transformer.transform(test_df)
        
        ## Train the model
        trainer = ModelTrainer()
        model, score = trainer.train(train_df, test_df)
        
        logger.info(f" Training completed. Best F1-score: {score}")
        print("MOdel training completed successfully.")
        print(f"Best F1-score: {score}")
        
        