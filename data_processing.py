import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Load dataset from CSV file"""
    try:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Dataset file not found at {file_path}")
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def handle_outliers(df):
    """Cap outliers using IQR method for all numeric columns"""
    try:
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if col.lower() not in ['id', 'quality']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = np.clip(df[col], lower_bound, upper_bound)
        logging.info("Outliers capped successfully")
        return df
    except Exception as e:
        logging.error(f"Error handling outliers: {str(e)}")
        raise

def save_processed_data(df, output_path):
    """Save processed data to CSV"""
    try:
        output_dir = Path(output_path).parent
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Processed data saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving processed data: {str(e)}")
        raise

def main():
    # Configuration
    RAW_DATA_PATH = "data/raw/WineQT.csv"
    PROCESSED_DATA_PATH = "data/processed/cleaned_wine_quality.csv"
    
    try:
        # Create directories
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        
        # Processing pipeline
        df = load_data(RAW_DATA_PATH)
        df_cleaned = handle_outliers(df)
        save_processed_data(df_cleaned, PROCESSED_DATA_PATH)
        
    except Exception as e:
        logging.critical(f"Data processing failed: {str(e)}")

if __name__ == "__main__":
    main()