# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
import pickle
import os

app = Flask(__name__)
CORS(app)  # Allow requests from your React Native app

# ── Load & train model (or load from pickle if already saved) ──
MODEL_PATH = "model.pkl"
ENCODER_PATH = "encoder.pkl"
SYMPTOMS_PATH = "symptoms.pkl"

def train_and_save():
    CSV_PATH = "fixed_augmented_dataset_multibiner_num_augmentations_100_cleaned.csv"
    df = pd.read_csv(CSV_PATH)
    TARGET_COL = "prognosis"

    SYMPTOM_COLS = [c for c in df.columns if c != TARGET_COL]

    # Augmentation for rare classes
    MIN_SAMPLES = 20
    augmented_rows = []
    for disease, count in df[TARGET_COL].value_counts().items():
        if count < MIN_SAMPLES:
            disease_rows = df[df[TARGET_COL] == disease]
            needed = MIN_SAMPLES - count
            for _ in range(needed):
                row = disease_rows.sample(1, replace=True).iloc[0].copy()
                flip_mask = np.random.random(len(SYMPTOM_COLS)) < 0.05
                for col, flip in zip(SYMPTOM_COLS, flip_mask):
                    if flip:
                        row[col] = 1 - row[col]
                augmented_rows.append(row)
    if augmented_rows:
        df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)

    X = df.drop(columns=[TARGET_COL]).values.astype(np.float32)
    y_raw = df[TARGET_COL].values
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    symptom_names = [c for c in df.columns if c != TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=30, min_samples_leaf=1,
        class_weight="balanced", random_state=42, n_jobs=1
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump(encoder, open(ENCODER_PATH, "wb"))
    pickle.dump(symptom_names, open(SYMPTOMS_PATH, "wb"))
    print("✅ Model trained and saved.")
    return model, encoder, symptom_names

# Load or train
if os.path.exists(MODEL_PATH):
    model = pickle.load(open(MODEL_PATH, "rb"))
    disease_encoder = pickle.load(open(ENCODER_PATH, "rb"))
    SYMPTOM_NAMES = pickle.load(open(SYMPTOMS_PATH, "rb"))
    print("✅ Loaded model from disk.")
else:
    model, disease_encoder, SYMPTOM_NAMES = train_and_save()


@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    """Returns the full list of valid symptom names."""
    return jsonify({"symptoms": sorted(SYMPTOM_NAMES)})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON: { "symptoms": ["fever", "cough", "fatigue"] }
    Returns top 5 predictions with confidence scores.
    """
    data = request.get_json()
    symptoms_list = data.get("symptoms", [])

    input_vec = np.zeros(len(SYMPTOM_NAMES), dtype=np.float32)
    unrecognised = []

    for sym in symptoms_list:
        sym_lower = sym.strip().lower()
        matched = [i for i, s in enumerate(SYMPTOM_NAMES) if s.lower() == sym_lower]
        if matched:
            input_vec[matched[0]] = 1
        else:
            unrecognised.append(sym)

    probabilities = model.predict_proba([input_vec])[0]
    top_indices = np.argsort(probabilities)[::-1][:5]

    results = []
    for idx in top_indices:
        results.append({
            "disease": disease_encoder.inverse_transform([idx])[0],
            "confidence": round(float(probabilities[idx]) * 100, 2)
        })

    return jsonify({
        "predictions": results,
        "unrecognised_symptoms": unrecognised,
        "low_confidence": bool(probabilities[top_indices[0]] < 0.30)
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)