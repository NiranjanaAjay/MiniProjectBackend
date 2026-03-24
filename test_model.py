import joblib
import pandas as pd

# Load trained model
model = joblib.load("xgboost_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

print("===== CureConnect Disease Prediction Demo =====")

# Take user input
age = int(input("Enter Age: "))
gender = int(input("Enter Gender (0 = Female, 1 = Male): "))
fever = int(input("Fever? (0/1): "))
cough = int(input("Cough? (0/1): "))
breathing_issue = int(input("Breathing Issue? (0/1): "))
fatigue = int(input("Fatigue? (0/1): "))
weight_loss = int(input("Weight Loss? (0/1): "))
systolic_bp = int(input("Systolic BP: "))
diastolic_bp = int(input("Diastolic BP: "))
cholesterol = int(input("Cholesterol: "))
hemoglobin = float(input("Hemoglobin: "))
wbc_count = int(input("WBC Count: "))
platelet_count = int(input("Platelet Count: "))
creatinine = float(input("Creatinine: "))
bilirubin = float(input("Bilirubin: "))

# Create dataframe
data = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "fever": fever,
    "cough": cough,
    "breathing_issue": breathing_issue,
    "fatigue": fatigue,
    "weight_loss": weight_loss,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "cholesterol": cholesterol,
    "hemoglobin": hemoglobin,
    "wbc_count": wbc_count,
    "platelet_count": platelet_count,
    "creatinine": creatinine,
    "bilirubin": bilirubin
}])

# Predict
prediction = model.predict(data)
disease = label_encoder.inverse_transform(prediction)

print("\n==============================")
print("Predicted Disease:", disease[0])
print("==============================")