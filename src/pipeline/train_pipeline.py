import pandas as pd
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logger

if __name__ == "__main__":

    logger.info("Starting training pipeline...")

    # Step 1: Ingest raw data — produces train.csv and test.csv.
    train_path, test_path = DataIngestion().initiate_data_ingestion()
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Step 2: Transform data.
    #
    # BUG FIX (Scaler refit) applied here at the pipeline level:
    # Previously, the code was:
    #   transformer = DataTransformation()
    #   train_df = transformer.transform(train_df)   # fit_transform on train
    #   test_df = transformer.transform(test_df)     # WRONG: fit_transform on test too
    #
    # Both calls used the same method name 'transform', which internally always
    # called fit_transform(). The test set got scaled with its own statistics.
    #
    # FIX: A single transformer instance is created. fit_transform() is called
    # on train data (learns and applies scaling), and transform() is called on
    # test data (applies training scaling only — no re-fitting).
    transformer = DataTransformation()
    train_df = transformer.fit_transform(train_df)   # learns scaler from train
    test_df = transformer.transform(test_df)         # applies train's scaler to test

    # Step 3: Train and evaluate models.
    trainer = ModelTrainer()
    model, score = trainer.train(train_df, test_df)

    logger.info(f"Training completed. Best weighted F1-score: {score:.4f}")
    print("Model training completed successfully.")
    print(f"Best F1-score: {score:.4f}")