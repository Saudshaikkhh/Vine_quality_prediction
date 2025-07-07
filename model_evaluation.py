import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_artifacts():
    """Load trained models and scaler"""
    try:
        scaler = joblib.load("models/scaler.pkl")
        models = {}
        for model_file in Path("models").glob("*.pkl"):
            if "scaler" not in model_file.name:
                model_name = model_file.stem
                models[model_name] = joblib.load(model_file)
        logging.info("Models and scaler loaded successfully")
        return models, scaler
    except Exception as e:
        logging.error(f"Error loading artifacts: {str(e)}")
        raise

def evaluate_model(model, X_test, X_test_scaled, y_test, model_name):
    """Evaluate model and generate reports"""
    try:
        # Make predictions
        if model_name == "RandomForest":
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Save report
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / f"{model_name}_report.txt"
        
        with open(report_path, "w") as f:
            f.write(f"{model_name} Evaluation Report\n")
            f.write("="*50 + "\n")
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\nConfusion Matrix:\n")
            f.write(np.array2string(cm))
        
        # Save confusion matrix plot
        fig_dir = report_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{model_name} Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(fig_dir / f"{model_name}_confusion_matrix.png")
        plt.close()
        
        logging.info(f"Evaluation completed for {model_name}")
        return accuracy
    except Exception as e:
        logging.error(f"Error evaluating {model_name}: {str(e)}")
        raise

def main(X_test, X_test_scaled, y_test):
    try:
        # Create reports directory
        Path("reports").mkdir(exist_ok=True)
        
        # Load artifacts
        models, scaler = load_artifacts()
        
        # Evaluate models
        results = {}
        for name, model in models.items():
            results[name] = evaluate_model(
                model, X_test, X_test_scaled, y_test, name
            )
        
        # Save summary report
        with open("reports/summary_report.txt", "w") as f:
            f.write("Model Performance Summary\n")
            f.write("="*50 + "\n")
            for model_name, accuracy in results.items():
                f.write(f"{model_name}: Accuracy = {accuracy:.4f}\n")
        
        logging.info("All models evaluated successfully")
    except Exception as e:
        logging.critical(f"Model evaluation failed: {str(e)}")

if __name__ == "__main__":
    # These would come from training pipeline
    import sys
    sys.path.append(".")
    from model_training import main as train_main
    _, X_test, y_test = train_main()
    main(X_test[0], X_test[1], y_test)