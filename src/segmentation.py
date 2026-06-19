import os
import sys
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# Add base directory to path so we can import src.config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def perform_segmentation():
    print("Loading cleaned data for segmentation...")
    df_clean = pd.read_csv(config.CLEAN_DATA_PATH)
    
    # Select behavioral features for clustering
    features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService', 'PaymentMethod']
    X_raw = df_clean[features].copy()
    
    # One-hot encode categorical features for richer clusters
    X_encoded = pd.get_dummies(X_raw, columns=['Contract', 'InternetService', 'PaymentMethod'], drop_first=True)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # Save the clustering scaler
    scaler_path = os.path.join(config.MODELS_DIR, 'segmentation_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Saved segmentation scaler to {scaler_path}")
    
    # Validate optimal K
    print("\nValidating K-Means with Silhouette Scores:")
    best_score = -1
    best_k = 2
    for k_test in range(2, 7):
        kmeans_test = KMeans(n_clusters=k_test, random_state=config.RANDOM_STATE, n_init=10)
        labels = kmeans_test.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"  K={k_test}: Silhouette Score = {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k_test
            
    print(f"Statistically optimal K is {best_k} (score: {best_score:.4f}). Applying K=4 for strategic 4-quadrant business consistency.")
    
    # Fit K-Means
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Save the K-Means model
    model_path = os.path.join(config.MODELS_DIR, 'kmeans_model.pkl')
    joblib.dump(kmeans, model_path)
    print(f"Saved K-Means model to {model_path}")
    
    df_clean['Cluster'] = clusters
    
    # Dynamically name the clusters based on centroids
    # Let's calculate the means for naming
    cluster_means = df_clean.groupby('Cluster')[['tenure', 'MonthlyCharges']].mean().reset_index()
    median_tenure = df_clean['tenure'].median()
    median_charges = df_clean['MonthlyCharges'].median()
    
    segment_map = {}
    for idx, row in cluster_means.iterrows():
        cluster_id = int(row['Cluster'])
        t_mean = row['tenure']
        c_mean = row['MonthlyCharges']
        
        if t_mean >= median_tenure and c_mean >= median_charges:
            segment_map[cluster_id] = "Loyal Premium"
        elif t_mean >= median_tenure and c_mean < median_charges:
            segment_map[cluster_id] = "Loyal Value"
        elif t_mean < median_tenure and c_mean >= median_charges:
            segment_map[cluster_id] = "High-Spend At-Risk"
        else:
            segment_map[cluster_id] = "New Budget"
            
    # Apply segment mapping
    df_clean['Segment'] = df_clean['Cluster'].map(segment_map)
    
    # Let's inspect the distribution
    print("\nCluster assignment counts:")
    print(df_clean['Segment'].value_counts())
    
    # Profile clusters: calculate metrics including Churn Rate
    # Convert Churn column back to binary for easy mean calculation
    df_clean['Churn_Numeric'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
    
    profile = df_clean.groupby('Segment').agg(
        Count=('customerID', 'count'),
        Avg_Tenure=('tenure', 'mean'),
        Avg_MonthlyCharges=('MonthlyCharges', 'mean'),
        Avg_TotalCharges=('TotalCharges', 'mean'),
        Churn_Rate=('Churn_Numeric', 'mean')
    ).reset_index()
    
    # Convert Churn_Rate to percentage
    profile['Churn_Rate'] = (profile['Churn_Rate'] * 100).round(2)
    profile['Avg_Tenure'] = profile['Avg_Tenure'].round(1)
    profile['Avg_MonthlyCharges'] = profile['Avg_MonthlyCharges'].round(2)
    profile['Avg_TotalCharges'] = profile['Avg_TotalCharges'].round(2)
    profile['Count'] = profile['Count'].astype(int)
    
    # Define strategic actions for each segment
    recommendations = {
        "Loyal Premium": "Focus on cross-selling and premium upgrades (e.g., streaming bundles). Offer loyalty rewards.",
        "Loyal Value": "Keep satisfied with consistent service quality. Offer multi-year contract discount options to lock in long-term.",
        "High-Spend At-Risk": "Proactively offer contract conversions from Month-to-Month to 1/2 Year with pricing discounts. Provide support checks.",
        "New Budget": "Smooth onboarding process. Offer entry-level utility upgrades. Send helpful tip sheets to increase feature usage."
    }
    
    profile['Strategic_Recommendation'] = profile['Segment'].map(recommendations)
    
    # Save profiles as JSON for dashboard
    profile_dict = profile.to_dict(orient='records')
    profile_path = os.path.join(config.MODELS_DIR, 'segment_profiles.json')
    with open(profile_path, 'w') as f:
        json.dump(profile_dict, f, indent=4)
    print(f"Saved segment profiles to {profile_path}")
    
    # Drop temp numeric churn column
    df_clean = df_clean.drop('Churn_Numeric', axis=1)
    
    # Save segmented data
    df_clean.to_csv(config.SEGMENTED_DATA_PATH, index=False)
    print(f"Saved segmented data to {config.SEGMENTED_DATA_PATH}")
    print("Segmentation completed successfully!")

if __name__ == "__main__":
    perform_segmentation()
