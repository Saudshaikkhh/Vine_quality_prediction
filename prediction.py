import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import warnings

# Silence that specific warning from sklearn about feature names mismatch
warnings.filterwarnings(
    "ignore",
    message=".*does not have valid feature names.*",
    category=UserWarning
)

# set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Map raw scores to human-friendly descriptors
QUALITY_LABELS = {
    3: "Below average",
    4: "Below average",
    5: "Average",
    6: "Above average",
    7: "High quality",
    8: "High quality",
    9: "Exceptional",
    10: "Exceptional"
}

class WineQualityPredictor:
    def __init__(self, models_dir="models"):
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.models_dir = Path(models_dir)
        self.load_models()
        
    def load_models(self):
        """Load all trained models and scaler from disk."""
        scaler_path = self.models_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        logging.info(f"Loaded scaler from {scaler_path}")

        # remember the feature order
        if hasattr(self.scaler, "feature_names_in_"):
            self.feature_names = list(self.scaler.feature_names_in_)
        else:
            raise AttributeError("Scaler missing feature_names_in_; retrain with a named DataFrame.")
        
        # load each model
        for pkl in self.models_dir.glob("*.pkl"):
            if pkl.stem.lower() == "scaler":
                continue
            name = pkl.stem
            self.models[name] = joblib.load(pkl)
            logging.info(f"Loaded model '{name}' from {pkl}")
        
        if not self.models:
            raise ValueError("No model files found in models directory.")
        logging.info("All models loaded successfully.")

    def preprocess_input(self, input_data):
        """Build DataFrame in the correct column order and scale it."""
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError("Input must be a dict or pandas DataFrame.")
        
        if 'Id' in df.columns:
            df = df.drop(columns=['Id'])
        
        missing = set(self.feature_names) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        df = df[self.feature_names]
        return self.scaler.transform(df)

    def predict(self, input_data, model_name="RandomForest"):
        """Return the raw numeric score."""
        if model_name not in self.models:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(self.models)}")
        X = self.preprocess_input(input_data)
        return int(self.models[model_name].predict(X)[0])

    def label_quality(self, score):
        """Lookup the human‑friendly label for a numeric score."""
        return QUALITY_LABELS.get(score, "Unknown")

if __name__ == "__main__":
    predictor = WineQualityPredictor(models_dir="models")
    
    sample_input = {
        'fixed acidity': 7.4,
        'volatile acidity': 0.7,
        'citric acid': 0.0,
        'residual sugar': 1.9,
        'chlorides': 0.076,
        'free sulfur dioxide': 11.0,
        'total sulfur dioxide': 34.0,
        'density': 0.9978,
        'pH': 3.51,
        'sulphates': 0.56,
        'alcohol': 9.4
    }
    
    try:
        for model in predictor.models:
            raw = predictor.predict(sample_input, model)
            label = predictor.label_quality(raw)
            print(f"{model}: {raw} → {label}")
    except Exception as err:
        print(f"Prediction failed: {err}")
