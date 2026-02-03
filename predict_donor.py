import pandas as pd
import pickle
import numpy as np

# Load the trained model
print("Loading model...")
with open('donor_model.pkl', 'rb') as f:
    saved_data = pickle.load(f)
    model = saved_data['model']
    label_encoders = saved_data['label_encoders']
    scaler = saved_data['scaler']
    feature_columns = saved_data['feature_columns']
    feature_means = saved_data['feature_means']

print("✓ Model loaded successfully!\n")

def predict_compatibility(donor_data):
    """
    Predict donor-patient compatibility score (0-100%)
    
    Parameters:
    -----------
    donor_data : dict
        Dictionary with keys:
        - HLA_match_score: float (7-10, e.g., 10 for "10/10 match")
        - donor_age: float (years)
        - recipient_age: float (years)
        - recipient_gender: str ("male" or "female")
        - CMV_status: str ("0", "1", "2", or "3" - compatibility level)
        - disease_group: str ("malignant" or "nonmalignant")
        - risk_group: str ("low" or "high")
    
    Returns:
    --------
    dict with 'score' (0-100%), 'status', and 'recommendation'
    """
    # Create DataFrame
    df = pd.DataFrame([donor_data])
    
    # Fill missing values with means
    for col in df.select_dtypes(include=['number']).columns:
        if col in feature_means:
            df[col] = df[col].fillna(feature_means[col])
    
    # Encode categorical variables
    for col in df.select_dtypes(include='object').columns:
        if col in label_encoders:
            le = label_encoders[col]
            # Handle unseen categories - use most common class
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
    
    # Ensure correct column order
    df = df[feature_columns]
    
    # Scale features
    df_scaled = scaler.transform(df)
    
    # Predict
    score = model.predict(df_scaled)[0]
    score = np.clip(score, 0, 100)  # Ensure 0-100 range
    
    # Determine status
    if score >= 80:
        status = "EXCELLENT"
        recommendation = "Highly recommended for transplant"
    elif score >= 60:
        status = "GOOD"
        recommendation = "Recommended for transplant"
    elif score >= 40:
        status = "FAIR"
        recommendation = "Consider with careful monitoring"
    else:
        status = "POOR"
        recommendation = "Not recommended - seek alternative donor"
    
    return {
        'score': score,
        'status': status,
        'recommendation': recommendation
    }

# Example predictions
print("="*70)
print("            DONOR-PATIENT COMPATIBILITY PREDICTOR")
print("="*70)

# Example 1: Excellent match - Perfect HLA, young donor, low risk
example1 = {
    'HLA_match_score': 10.0,
    'donor_age': 25.0,
    'recipient_age': 8.0,
    'recipient_gender': 'male',
    'CMV_status': '0',  # Best compatibility
    'disease_group': 'nonmalignant',
    'risk_group': 'low'
}

result1 = predict_compatibility(example1)
print("\n--- Example 1: Optimal Match ---")
print(f"HLA Match       : 10/10 (Perfect)")
print(f"Donor Age       : {example1['donor_age']:.0f} years")
print(f"Recipient Age   : {example1['recipient_age']:.0f} years")
print(f"CMV Status      : {example1['CMV_status']} (matched)")
print(f"Disease Group   : {example1['disease_group']}")
print(f"Risk Group      : {example1['risk_group']}")
print(f"\n→ Compatibility : {result1['score']:.1f}%")
print(f"→ Status        : {result1['status']}")
print(f"→ Recommendation: {result1['recommendation']}")

# Example 2: Good match - 9/10 HLA, moderate age
example2 = {
    'HLA_match_score': 9.0,
    'donor_age': 35.0,
    'recipient_age': 12.0,
    'recipient_gender': 'female',
    'CMV_status': '1',
    'disease_group': 'malignant',
    'risk_group': 'low'
}

result2 = predict_compatibility(example2)
print("\n--- Example 2: Good Match ---")
print(f"HLA Match       : 9/10")
print(f"Donor Age       : {example2['donor_age']:.0f} years")
print(f"Recipient Age   : {example2['recipient_age']:.0f} years")
print(f"CMV Status      : {example2['CMV_status']}")
print(f"Disease Group   : {example2['disease_group']}")
print(f"Risk Group      : {example2['risk_group']}")
print(f"\n→ Compatibility : {result2['score']:.1f}%")
print(f"→ Status        : {result2['status']}")
print(f"→ Recommendation: {result2['recommendation']}")

# Example 3: Poor match - Low HLA, high risk
example3 = {
    'HLA_match_score': 7.0,
    'donor_age': 45.0,
    'recipient_age': 18.0,
    'recipient_gender': 'male',
    'CMV_status': '3',  # Worst compatibility
    'disease_group': 'malignant',
    'risk_group': 'high'
}

result3 = predict_compatibility(example3)
print("\n--- Example 3: Marginal Match ---")
print(f"HLA Match       : 7/10")
print(f"Donor Age       : {example3['donor_age']:.0f} years")
print(f"Recipient Age   : {example3['recipient_age']:.0f} years")
print(f"CMV Status      : {example3['CMV_status']} (mismatched)")
print(f"Disease Group   : {example3['disease_group']}")
print(f"Risk Group      : {example3['risk_group']}")
print(f"\n→ Compatibility : {result3['score']:.1f}%")
print(f"→ Status        : {result3['status']}")
print(f"→ Recommendation: {result3['recommendation']}")

print("\n" + "="*70)
print("\n✓ Ready for custom predictions!")
print("\nTo test your own data, modify the example dictionaries above or")
print("call predict_compatibility() with your donor-patient data.")
print("\nValid values:")
print("  - HLA_match_score: 7.0, 8.0, 9.0, or 10.0")
print("  - recipient_gender: 'male' or 'female'")
print("  - CMV_status: '0', '1', '2', '3' (0=best, 3=worst)")
print("  - disease_group: 'malignant' or 'nonmalignant'")
print("  - risk_group: 'low' or 'high'")
print("="*70)