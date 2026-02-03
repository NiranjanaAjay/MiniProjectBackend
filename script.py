import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Load Data
df = pd.read_csv("bone_marrow_syn.csv")

# Define target (using survival_status as proxy for score, or your custom score)
TARGET = "survival_status" 
df[TARGET] = df[TARGET] * 100  # Scaling 0-1 to 0-100 for your logic

# 2. Preprocessing & Encoding
label_encoders = {}
categorical_cols = ['recipient_gender', 'CMV_status', 'disease_group', 'risk_group']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Map HLA_match string (10/10) to float (10.0) if needed
if df['HLA_match'].dtype == 'object':
    df['HLA_match_score'] = df['HLA_match'].str.extract(r'(\d+)').astype(float)
else:
    df['HLA_match_score'] = df['HLA_match']

# Select features used in your predictor script
feature_columns = ['HLA_match_score', 'donor_age', 'recipient_age', 
                   'recipient_gender', 'CMV_status', 'disease_group', 'risk_group']

X = df[feature_columns]
y = df[TARGET]

# 3. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_scaled, y)

# 5. SAVE EVERYTHING TO PICKLE
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

print("✓ donor_model.pkl has been generated!")