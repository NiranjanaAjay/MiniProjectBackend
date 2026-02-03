import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- 1. Load Data ---
file_path = "donor_dataset.csv"
if not os.path.exists(file_path):
    print(f"❌ Error: {file_path} not found.")
    exit()

df = pd.read_csv(file_path)

# --- 2. Define Target ---
TARGET = "survival_status" 
# Convert 0/1 to 0-100 scale for your predictor script logic
df[TARGET] = df[TARGET] * 100 

# --- 3. Feature Selection (THE IMPORTANT PART) ---
# We REMOVE survival_time because it's "cheating" (Data Leakage)
feature_columns = [
    'donor_age', 'recipient_age', 'recipient_gender', 
    'CMV_status', 'disease_group', 'risk_group'
]

# Add HLA Match Score extraction
if 'HLA_match' in df.columns:
    df['HLA_match_score'] = df['HLA_match'].astype(str).str.extract(r'(\d+)').astype(float)
    feature_columns.append('HLA_match_score')

# --- 4. Encoding ---
label_encoders = {}
categorical_cols = ['recipient_gender', 'CMV_status', 'disease_group', 'risk_group']

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

X = df[feature_columns]
y = df[TARGET]

# --- 5. Train & Scale ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_scaled, y)

# --- 6. Save Pickle ---
feature_means = X.mean().to_dict()
saved_data = {
    'model': model,
    'label_encoders': label_encoders,
    'scaler': scaler,
    'feature_columns': feature_columns,
    'feature_means': feature_means
}

with open('donor_model.pkl', 'wb') as f:
    pickle.dump(saved_data, f)

print("✅ SUCCESS: 'donor_model.pkl' generated without 'cheating' features.")

# --- 7. Check Feature Importance ---
importances = model.feature_importances_
print("\n--- Real Feature Importance ---")
for name, imp in sorted(zip(feature_columns, importances), key=lambda x: x[1], reverse=True):
    print(f"{name:20}: {round(imp*100, 2)}%")