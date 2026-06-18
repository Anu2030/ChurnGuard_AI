import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'telco.csv')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Processed file paths
CLEAN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'telco_clean.csv')
PREPROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'telco_preprocessed.csv')
SEGMENTED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'telco_segmented.csv')
CLTV_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'telco_cltv.csv')

# Model output paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

# Feature definitions
TARGET_COL = 'Churn'
ID_COL = 'customerID'

# Numeric columns to scale
NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Categorical columns to encode
CATEGORICAL_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# Random state for reproducibility
RANDOM_STATE = 42

# Hyperparameter grids for training
GRID_PARAMS_RF = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

GRID_PARAMS_XGB = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}
