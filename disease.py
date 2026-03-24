import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# 1️⃣ Load dataset
df = pd.read_csv("careconnect_synthetic_dataset.csv")

# 2️⃣ Separate features and target
X = df.drop("target_disease", axis=1)
y = df["target_disease"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 3️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 4️⃣ Create model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# 5️⃣ Train
model.fit(X_train, y_train)

# 6️⃣ Predict
y_pred = model.predict(X_test)

# 7️⃣ Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 8️⃣ Save model and encoder
joblib.dump(model, "xgboost_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nModel saved as xgboost_model.pkl")