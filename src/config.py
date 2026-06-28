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
TRANSACTIONS_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'synthetic_transactions.csv')
RFM_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'telco_rfm.csv')

# Model output paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

# Feature definitions
TARGET_COL = 'Churn'
ID_COL = 'customerID'

# Numeric columns to scale
NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TenureInYears', 'ChargeRatio']

# Categorical columns to encode
CATEGORICAL_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# Random state for reproducibility
RANDOM_STATE = 42

# Hyperparameter grids for training (balanced: deep enough to be rigorous, fast enough to run)
GRID_PARAMS_RF = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

GRID_PARAMS_XGB = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

GRID_PARAMS_GB = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}
