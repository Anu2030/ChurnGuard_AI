import os
import sys
import pandas as pd
from datetime import datetime

# Add base directory to path so we can import src.config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def perform_rfm_analysis():
    print("Loading synthetic transaction data for RFM analysis...")
    if not os.path.exists(config.TRANSACTIONS_DATA_PATH):
        print(f"Error: Transactions file not found at {config.TRANSACTIONS_DATA_PATH}")
        print("Please run generate_transactions.py first.")
        return
        
    df_txns = pd.read_csv(config.TRANSACTIONS_DATA_PATH)
    
    # Convert string to datetime
    df_txns['transaction_date'] = pd.to_datetime(df_txns['transaction_date'])
    
    # Define "current date" as the day after the most recent transaction in the dataset
    current_date = df_txns['transaction_date'].max() + pd.Timedelta(days=1)
    
    # Calculate R, F, M
    rfm = df_txns.groupby('customerID').agg(
        Recency=('transaction_date', lambda x: (current_date - x.max()).days),
        Frequency=('customerID', 'count'),
        Monetary=('amount', 'sum')
    ).reset_index()
    
    # Scoring 1 to 5 using quintiles (qcut)
    # For Recency: lower days is better, so labels are 5 (best) down to 1 (worst)
    # Using duplicates='drop' in case many users have exactly 1 frequency
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(int)
    
    # For Frequency and Monetary: higher is better, so labels 1 to 5
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)
    
    # Function to map scores to buckets
    def map_rfm_segment(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        
        # Champions: R(4-5), F(4-5), M(4-5)
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
            
        # At Risk: R(1-2), F(3-5), M(3-5)
        elif r <= 2 and f >= 3 and m >= 3:
            return "At Risk"
            
        # Hibernating: R(1-2), F(1-2), M(1-2)
        elif r <= 2 and f <= 2 and m <= 2:
            return "Hibernating"
            
        # Potential Loyalists: R(4-5), F(1-3), M(1-3)
        elif r >= 4 and f <= 3 and m <= 3:
            return "Potential Loyalists"
            
        # Loyal Customers: R(3-5), F(3-5), M(1-3)
        elif r >= 3 and f >= 3 and m <= 3:
            return "Loyal Customers"
            
        # Catch-all for others
        else:
            return "Needs Attention"
            
    rfm['RFM_Segment'] = rfm.apply(map_rfm_segment, axis=1)
    
    # Merge back some core customer details from clean_data for richer dashboard filtering
    df_clean = pd.read_csv(config.CLEAN_DATA_PATH)[['customerID', 'Churn', 'Contract']]
    rfm_final = pd.merge(rfm, df_clean, on='customerID', how='inner')
    
    # Save the RFM results
    rfm_final.to_csv(config.RFM_DATA_PATH, index=False)
    
    print("\nRFM Segmentation Distribution:")
    print(rfm_final['RFM_Segment'].value_counts())
    
    print(f"\nSaved RFM Analysis to {config.RFM_DATA_PATH}")

if __name__ == "__main__":
    perform_rfm_analysis()
