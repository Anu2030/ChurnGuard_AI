import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import sys

# Add base directory to path so we can import src.config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def clean_data(df):
    """
    Cleans raw Telco Churn dataset:
    - Coerces TotalCharges to numeric, handles missing values.
    - Handles SeniorCitizen data representation.
    """
    df = df.copy()
    
    # Clean TotalCharges: replace spaces with NaN, coerce to float
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
    
    # Fill missing TotalCharges: since these are new customers (tenure=0), TotalCharges can be 0 or equal to MonthlyCharges
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
    
    # Feature Engineering
    df['TenureInYears'] = df['tenure'] / 12.0
    df['ChargeRatio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 0.001)
    
    # SeniorCitizen is 0 or 1, let's convert to string Yes/No for cleaner EDA/Segmentation, 
    # but we'll encode it back in preprocessing
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    return df

def preprocess_for_ml(df):
    """
    Preprocesses cleaned data for machine learning:
    - Target column encoding (Yes: 1, No: 0)
    - One-hot encoding of categorical variables
    - Scaling numeric features
    - Saving scaler object
    """
    df = df.copy()
    
    # Save original customer ID if it exists, otherwise ignore
    customer_ids = df[config.ID_COL] if config.ID_COL in df.columns else None
    
    # Target encoding
    if config.TARGET_COL in df.columns:
        df[config.TARGET_COL] = df[config.TARGET_COL].map({'Yes': 1, 'No': 0})
    
    # Drop customer ID
    if config.ID_COL in df.columns:
        df = df.drop(config.ID_COL, axis=1)
        
    # Get categorical and numerical columns dynamically to match config
    cat_cols = [c for c in config.CATEGORICAL_COLS if c in df.columns and c != config.TARGET_COL]
    num_cols = [c for c in config.NUMERIC_COLS if c in df.columns]
    
    # One-hot encode all categorical columns
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Ensure all boolean columns created by get_dummies are converted to 1/0
    bool_cols = df_encoded.select_dtypes(include='bool').columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    
    # Fit and save scaler on numeric columns
    scaler = StandardScaler()
    df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])
    
    # Make sure target column is the last column if it exists
    if config.TARGET_COL in df_encoded.columns:
        target = df_encoded[config.TARGET_COL]
        df_encoded = df_encoded.drop(config.TARGET_COL, axis=1)
        df_encoded[config.TARGET_COL] = target
        
    # Ensure models directory exists
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Save the scaler
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"Saved scaler to {config.SCALER_PATH}")
    
    return df_encoded

def main():
    print("Starting Preprocessing Pipeline...")
    
    # Ensure raw file exists
    if not os.path.exists(config.RAW_DATA_PATH):
        print(f"Error: Raw data file not found at {config.RAW_DATA_PATH}")
        sys.exit(1)
        
    # Load raw data
    df_raw = pd.read_csv(config.RAW_DATA_PATH)
    print(f"Loaded raw data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")
    
    # 1. Clean Data (for EDA / Segmentation)
    df_clean = clean_data(df_raw)
    
    # Ensure processed directory exists
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    
    # Save clean data
    df_clean.to_csv(config.CLEAN_DATA_PATH, index=False)
    print(f"Cleaned data saved to {config.CLEAN_DATA_PATH}")
    
    # 2. Preprocess Data (for classification models)
    df_ml = preprocess_for_ml(df_clean)
    
    # Save preprocessed data
    df_ml.to_csv(config.PREPROCESSED_DATA_PATH, index=False)
    print(f"Preprocessed data saved to {config.PREPROCESSED_DATA_PATH}")
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()