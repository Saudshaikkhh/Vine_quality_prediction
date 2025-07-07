# Wine Quality Prediction System

A comprehensive machine learning pipeline for predicting wine quality using physicochemical properties. This system implements multiple classification algorithms to assess wine quality on a scale from 3 to 10, with human-readable quality descriptors.

## Overview

The Wine Quality Prediction System is designed to analyze wine samples based on their chemical composition and predict their quality rating. The system processes raw wine data, trains multiple machine learning models, and provides an easy-to-use prediction interface with quality labels ranging from "Below average" to "Exceptional."

## Features

- **Automated Data Processing**: Handles outliers using the Interquartile Range (IQR) method
- **Multiple Model Training**: Implements Random Forest, SGD Classifier, and Support Vector Machine
- **Comprehensive Evaluation**: Generates detailed performance reports with confusion matrices
- **Quality Mapping**: Converts numeric scores to descriptive quality labels
- **Robust Error Handling**: Comprehensive logging and exception management
- **Scalable Architecture**: Modular design for easy maintenance and extension

## System Architecture

The system consists of four main components:

### Data Processing (`data_processing.py`)
- Loads raw wine quality dataset
- Handles outliers using statistical methods
- Saves cleaned data for model training
- Implements comprehensive error handling and logging

### Model Training (`model_training.py`)
- Trains multiple classification models simultaneously
- Applies feature scaling using StandardScaler
- Splits data into training and testing sets
- Saves trained models and preprocessing artifacts

### Model Evaluation (`model_evaluation.py`)
- Evaluates all trained models on test data
- Generates detailed classification reports
- Creates confusion matrix visualizations
- Produces summary performance comparisons

### Prediction Interface (`prediction.py`)
- Provides a user-friendly prediction interface
- Loads trained models and preprocessing components
- Maps numeric predictions to quality descriptors
- Handles various input formats (dict, DataFrame)

## Installation

Ensure you have Python 3.7+ installed, then install the required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib pathlib
```

## Usage

### Data Processing
```python
python data_processing.py
```

### Model Training
```python
python model_training.py
```

### Model Evaluation
```python
python model_evaluation.py
```

### Making Predictions
```python
from prediction import WineQualityPredictor

# Initialize predictor
predictor = WineQualityPredictor()

# Sample wine data
wine_sample = {
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

# Get prediction
quality_score = predictor.predict(wine_sample)
quality_label = predictor.label_quality(quality_score)
print(f"Wine Quality: {quality_score} - {quality_label}")
```

## Input Features

The system requires the following 11 physicochemical properties:

- **Fixed Acidity**: Tartaric acid concentration
- **Volatile Acidity**: Acetic acid concentration
- **Citric Acid**: Citric acid concentration
- **Residual Sugar**: Sugar remaining after fermentation
- **Chlorides**: Salt concentration
- **Free Sulfur Dioxide**: Free SO2 concentration
- **Total Sulfur Dioxide**: Total SO2 concentration
- **Density**: Wine density
- **pH**: Acidity level
- **Sulphates**: Potassium sulfate concentration
- **Alcohol**: Alcohol percentage

## Quality Scale

The system predicts wine quality on a scale from 3 to 10, mapped to descriptive labels:

- **3-4**: Below average
- **5**: Average
- **6**: Above average
- **7-8**: High quality
- **9-10**: Exceptional

## Directory Structure

```
wine_quality_prediction/
├── data/
│   ├── raw/
│   │   └── WineQT.csv
│   └── processed/
│       └── cleaned_wine_quality.csv
├── models/
│   ├── RandomForest.pkl
│   ├── SGDClassifier.pkl
│   ├── SVC.pkl
│   └── scaler.pkl
├── reports/
│   ├── figures/
│   └── summary_report.txt
├── data_processing.py
├── model_training.py
├── model_evaluation.py
└── prediction.py
```

## Model Performance

The system trains three different algorithms:

- **Random Forest**: Ensemble method for robust predictions
- **SGD Classifier**: Stochastic Gradient Descent for linear classification
- **Support Vector Machine**: Kernel-based classification

Each model is evaluated using accuracy metrics, classification reports, and confusion matrices to ensure reliable performance.

## Technical Implementation

### Data Preprocessing
- Outlier detection and capping using IQR method
- Feature scaling with StandardScaler
- Proper handling of missing values and data types

### Model Training
- Cross-validation ready architecture
- Consistent random state for reproducibility
- Separate preprocessing pipelines for different model types

### Evaluation Framework
- Comprehensive metrics calculation
- Visual performance analysis
- Automated report generation

## Error Handling

The system implements robust error handling throughout:
- File existence validation
- Data format verification
- Model loading safeguards
- Graceful failure recovery
- Comprehensive logging system

## Future Enhancements

Potential improvements include:
- Hyperparameter tuning with grid search
- Cross-validation implementation
- Feature importance analysis
- Model ensemble methods
- Real-time prediction API
- Web interface development

## Dependencies

- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning algorithms
- `matplotlib`: Static plotting
- `seaborn`: Statistical visualization
- `joblib`: Model persistence
- `pathlib`: Path handling

## License

This project is available under the MIT License, allowing for both personal and commercial use with proper attribution.
