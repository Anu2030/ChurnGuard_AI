import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add base directory to path so we can import src.config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def generate_transactions():
    print("Loading cleaned data to generate synthetic transactions...")
    df_clean = pd.read_csv(config.CLEAN_DATA_PATH)
    
    # We will anchor our "current date" to a fixed point for reproducibility
    current_date = datetime(2023, 12, 31)
    
    transactions = []
    
    # Set seed for reproducibility
    np.random.seed(config.RANDOM_STATE)
    
    for _, row in df_clean.iterrows():
        cust_id = row['customerID']
        tenure = int(row['tenure'])
        monthly_charge = float(row['MonthlyCharges'])
        churned = row['Churn'] == 'Yes'
        
        # Determine the date of the LAST transaction
        if churned:
            # Churned customers stopped buying some time ago (e.g. 30 to 180 days ago)
            days_ago = np.random.randint(30, 180)
        else:
            # Active customers bought recently (e.g. 1 to 30 days ago)
            days_ago = np.random.randint(1, 30)
            
        last_txn_date = current_date - timedelta(days=days_ago)
        
        # If tenure is 0, they just signed up and haven't stayed a full month. 
        # Give them exactly 1 transaction (the signup payment).
        if tenure == 0:
            num_txns = 1
        else:
            # Generally, people pay once a month, so frequency is roughly equal to tenure.
            # We add slight random variation (-1 to +2) to make it realistic.
            num_txns = max(1, tenure + np.random.randint(-1, 3))
            
        # Generate the historical transactions working backwards from the last_txn_date
        for i in range(num_txns):
            # Transactions spaced roughly 30 days apart
            txn_date = last_txn_date - timedelta(days=(i * 30) + np.random.randint(-3, 4))
            
            # Amount is roughly the monthly charge, with minor fluctuations (e.g. overage charges)
            # Normal distribution centered at monthly_charge with standard dev of 5% of charge
            txn_amount = np.random.normal(loc=monthly_charge, scale=monthly_charge * 0.05)
            txn_amount = round(max(5.0, txn_amount), 2) # ensure positive
            
            transactions.append({
                'customerID': cust_id,
                'transaction_date': txn_date.strftime('%Y-%m-%d'),
                'amount': txn_amount
            })
            
    df_txns = pd.DataFrame(transactions)
    
    # Sort chronologically
    df_txns = df_txns.sort_values(by=['transaction_date', 'customerID'])
    
    # Save to CSV
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df_txns.to_csv(config.TRANSACTIONS_DATA_PATH, index=False)
    print(f"Generated {len(df_txns)} synthetic transactions for {len(df_clean)} customers.")
    print(f"Saved to {config.TRANSACTIONS_DATA_PATH}")

if __name__ == "__main__":
    generate_transactions()
