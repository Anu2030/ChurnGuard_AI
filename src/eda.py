import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r'C:\Users\HP\churn_cltv\data\raw\telco.csv')

# Clean TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

print("Shape:", df.shape)
print("\nOverall churn rate:")
print(df['Churn'].value_counts(normalize=True) * 100)

# --- Plot 1: Churn by Contract Type ---
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title('Churn by Contract Type')
plt.savefig('notebooks/churn_by_contract.png')
plt.close()

# --- Plot 2: Tenure distribution by Churn ---
plt.figure(figsize=(7, 5))
sns.histplot(data=df, x='tenure', hue='Churn', bins=30, multiple='stack')
plt.title('Tenure Distribution by Churn')
plt.savefig('notebooks/tenure_by_churn.png')
plt.close()

# --- Plot 3: Monthly Charges by Churn ---
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
plt.title('Monthly Charges by Churn')
plt.savefig('notebooks/monthlycharges_by_churn.png')
plt.close()

print("\n3 plots saved in notebooks/ folder")

# Save cleaned data for next phase
df.to_csv('data/processed/telco_clean.csv', index=False)
print("Cleaned data saved to data/processed/telco_clean.csv")