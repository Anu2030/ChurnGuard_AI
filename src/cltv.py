import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
from lifelines import CoxPHFitter, KaplanMeierFitter

# Add base directory to path so we can import src.config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def calculate_cltv():
    print("Loading segmented customer data...")
    df = pd.read_csv(config.SEGMENTED_DATA_PATH)
    
    # 1. Prepare data for survival analysis
    df_survival = df.copy()
    
    # Target column: Churn (Yes: 1, No: 0)
    df_survival['Churn_Event'] = df_survival['Churn'].map({'Yes': 1, 'No': 0})
    
    # Ensure tenure is float and > 0 (Cox PH requires positive duration)
    df_survival['tenure'] = df_survival['tenure'].astype(float)
    df_survival.loc[df_survival['tenure'] == 0, 'tenure'] = 0.5  # Adjust 0 tenure to 0.5 months
    
    # Select covariates for Cox Proportional Hazards Model
    covariates = [
        'tenure', 'Churn_Event', 'MonthlyCharges', 
        'Contract', 'InternetService', 'PaymentMethod', 'PaperlessBilling'
    ]
    df_cox_input = df_survival[covariates].copy()
    
    # One-hot encode categorical features for lifelines
    cat_cols = ['Contract', 'InternetService', 'PaymentMethod', 'PaperlessBilling']
    df_cox_input = pd.get_dummies(df_cox_input, columns=cat_cols, drop_first=True)
    
    # Ensure all boolean columns are converted to 1/0
    bool_cols = df_cox_input.select_dtypes(include='bool').columns
    df_cox_input[bool_cols] = df_cox_input[bool_cols].astype(int)
    
    # 2. Fit Cox Proportional Hazards Model
    print("Fitting Cox Proportional Hazards Model...")
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_cox_input, duration_col='tenure', event_col='Churn_Event')
    
    # Print model summary
    print("\nCox PH Model Summary:")
    print(cph.summary[['coef', 'exp(coef)', 'p']])
    
    # Save the fitted Cox model
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(cph, os.path.join(config.MODELS_DIR, 'cox_model.pkl'))
    print(f"Saved Cox model to models/cox_model.pkl")
    
    # 3. Predict survival probabilities and calculate Expected Remaining Tenure
    print("\nPredicting remaining tenure for active customers...")
    
    # Generate conditional survival curves for each customer
    # We will project up to 72 months (typical maximum tenure in Telco dataset is 72)
    max_projection_months = 72
    
    # Get survival function curves for all customers
    # rows: months (0 to 72), columns: customer index
    survival_curves = cph.predict_survival_function(df_cox_input)
    
    expected_remaining_tenure = []
    
    for i in range(len(df_survival)):
        current_tenure = df_survival.loc[i, 'tenure']
        is_churned = df_survival.loc[i, 'Churn_Event'] == 1
        
        if is_churned:
            # Churned customers have 0 remaining tenure
            expected_remaining_tenure.append(0.0)
        else:
            # For active customers, calculate expected remaining months
            # Sum of conditional survival probabilities S(t | t > T) = S(t) / S(T) for t > T
            # Get index of current tenure in survival curves (using closest value)
            current_t_idx = min(survival_curves.index, key=lambda x: abs(x - current_tenure))
            s_current = survival_curves.loc[current_t_idx, i]
            
            if s_current == 0:
                expected_remaining_tenure.append(0.0)
                continue
                
            # Filter survival curve for times greater than current tenure
            future_times = [t for t in survival_curves.index if t > current_tenure and t <= max_projection_months]
            
            if len(future_times) == 0:
                # If they are already at the max tenure, assume a baseline remaining tenure
                expected_remaining_tenure.append(6.0)  # Assume average 6 more months
            else:
                s_future = survival_curves.loc[future_times, i]
                # Expected remaining tenure is the sum of conditional survival probabilities
                cond_surv = s_future / s_current
                # Integrate using trapezoidal rule or simple sum
                remaining_months = cond_surv.sum()
                expected_remaining_tenure.append(remaining_months)
                
    df['Expected_Remaining_Tenure'] = expected_remaining_tenure
    df['Expected_Total_Tenure'] = df['tenure'] + df['Expected_Remaining_Tenure']
    
    # 4. Calculate Customer Lifetime Value (CLTV)
    # Standard formula: CLTV = Expected Total Tenure * Monthly Charges * Gross Profit Margin
    # Let's assume a standard telecom Gross Profit Margin of 70% (0.70)
    profit_margin = 0.70
    df['CLTV'] = df['Expected_Total_Tenure'] * df['MonthlyCharges'] * profit_margin
    df['CLTV'] = df['CLTV'].round(2)
    
    # 5. Segment CLTV into tiers (Low, Medium, High Value)
    df['CLTV_Level'] = pd.qcut(df['CLTV'], q=3, labels=['Low Value', 'Medium Value', 'High Value'])
    
    print("\nCLTV Level distribution:")
    print(df['CLTV_Level'].value_counts())
    
    # Save the updated dataset
    df.to_csv(config.CLTV_DATA_PATH, index=False)
    print(f"Saved CLTV results to {config.CLTV_DATA_PATH}")
    
    # 6. Fit Kaplan-Meier for cohort baseline curves to be plotted in dashboard
    kmf = KaplanMeierFitter()
    
    # Save KM fit data by Contract cohort for interactive frontend plotting
    cohorts_km_data = {}
    for contract_type in df['Contract'].unique():
        idx = (df['Contract'] == contract_type)
        kmf.fit(df_survival.loc[idx, 'tenure'], event_observed=df_survival.loc[idx, 'Churn_Event'])
        cohorts_km_data[contract_type] = {
            'timeline': kmf.survival_function_.index.tolist(),
            'survival_probability': kmf.survival_function_['KM_estimate'].tolist()
        }
        
    km_data_path = os.path.join(config.MODELS_DIR, 'km_cohorts_data.json')
    with open(km_data_path, 'w') as f:
        json.dump(cohorts_km_data, f)
    print(f"Saved cohort survival curves to {km_data_path}")
    print("CLTV Modeling completed successfully!")

if __name__ == "__main__":
    calculate_cltv()
