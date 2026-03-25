import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logger


class DataIngestion:
    def __init__(self):
        # For Option A (binary, single file):
        # Set raw_data_path to the single Friday DDoS file.
        #
        # For Option C (multi-class, full week):
        # Set raw_data_path to a directory containing all daily CSVs,
        # OR set raw_data_paths to a list of file paths (see below).
        # The class handles both modes transparently.
        self.raw_data_path = "data/raw/Friday_DDos.csv"

        # Option C: provide a list of daily CSVs to concatenate for multi-class.
        # Example:
        #   self.raw_data_paths = [
        #       "data/raw/Monday-WorkingHours.pcap_ISCX.csv",
        #       "data/raw/Tuesday-WorkingHours.pcap_ISCX.csv",
        #       "data/raw/Wednesday-workingHours.pcap_ISCX.csv",
        #       "data/raw/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        #       "data/raw/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        #       "data/raw/Friday-WorkingHours-Morning.pcap_ISCX.csv",
        #       "data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        #   ]
        # Leave as None to use single-file mode.
        self.raw_data_paths = None

        self.train_data_path = "data/processed/train.csv"
        self.test_data_path = "data/processed/test.csv"

    def _load_raw_data(self) -> pd.DataFrame:
        """
        Internal helper. Loads one file or concatenates multiple files.
        Strips leading/trailing whitespace from column names immediately on load
        so downstream code doesn't need to worry about space-prefixed column names.
        """
        if self.raw_data_paths is not None:
            # Multi-file mode (Option C): load all CSVs and concatenate.
            frames = []
            for path in self.raw_data_paths:
                logger.info(f"Loading file: {path}")
                df = pd.read_csv(path, low_memory=False)
                df.columns = df.columns.str.strip()
                frames.append(df)
            df = pd.concat(frames, ignore_index=True)
            logger.info(f"Concatenated {len(self.raw_data_paths)} files. Total rows: {len(df)}")
        else:
            # Single-file mode (Option A / binary).
            logger.info(f"Loading file: {self.raw_data_path}")
            df = pd.read_csv(self.raw_data_path, low_memory=False)
            df.columns = df.columns.str.strip()

        return df

    def initiate_data_ingestion(self):
        logger.info("Data Ingestion started.")

        try:
            # Skip re-ingestion if processed files already exist.
            # This saves time during iterative development.
            if os.path.exists(self.train_data_path) and os.path.exists(self.test_data_path):
                logger.info("Train and Test data already exist. Skipping ingestion.")
                return self.train_data_path, self.test_data_path

            df = self._load_raw_data()
            logger.info(f"Dataset loaded. Shape: {df.shape}")

            # Log label distribution — important for understanding class imbalance,
            # which is a key issue in CIC-IDS2017 (and relevant to your paper).
            label_col = 'Label'
            if label_col in df.columns:
                logger.info(f"Label distribution:\n{df[label_col].value_counts()}")

            processed_dir = os.path.dirname(self.train_data_path)
            os.makedirs(processed_dir, exist_ok=True)

            # Stratified split: preserves class proportions in both train and test.
            # This is especially critical for multi-class (Option C) where some
            # attack types have very few samples.
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df[label_col]
            )

            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)

            logger.info(
                f"Train/test split saved. "
                f"Train: {len(train_set)} rows, Test: {len(test_set)} rows."
            )
            return self.train_data_path, self.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()