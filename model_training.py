import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_processed_data(file_path):
    """Load processed dataset"""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Processed data loaded with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading processed data: {str(e)}")
        raise

def prepare_data(df):
    """Split data into features and target"""
    try:
        X = df.drop(columns=['quality', 'Id'])
        y = df['quality'].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logging.info("Data split into train/test sets")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error preparing data: {str(e)}")
        raise

def train_models(X_train, X_test, y_train):
    """Train and return multiple classifiers"""
    try:
        # Initialize models
        models = {
            "RandomForest": RandomForestClassifier(random_state=42),
            "SGDClassifier": SGDClassifier(random_state=42),
            "SVC": SVC(random_state=42)
        }
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        trained_models = {}
        for name, model in models.items():
            if name == "RandomForest":
                model.fit(X_train, y_train)
            else:
                model.fit(X_train_scaled, y_train)
            trained_models[name] = model
            logging.info(f"{name} trained successfully")
        
        return trained_models, scaler, X_test_scaled
    except Exception as e:
        logging.error(f"Error training models: {str(e)}")
        raise

def save_artifacts(models, scaler):
    """Save trained models and scaler"""
    try:
        Path("models").mkdir(exist_ok=True)
        joblib.dump(scaler, "models/scaler.pkl")
        for name, model in models.items():
            joblib.dump(model, f"models/{name}.pkl")
        logging.info("Models and scaler saved successfully")
    except Exception as e:
        logging.error(f"Error saving artifacts: {str(e)}")
        raise

def main():
    try:
        # Load processed data
        df = load_processed_data("data/processed/cleaned_wine_quality.csv")
        
        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data(df)
        
        # Train models
        models, scaler, X_test_scaled = train_models(X_train, X_test, y_train)
        
        # Save artifacts
        save_artifacts(models, scaler)
        
        return X_test, X_test_scaled, y_test
    except Exception as e:
        logging.critical(f"Model training failed: {str(e)}")

if __name__ == "__main__":
    main()