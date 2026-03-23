# ============================================================
# api.py
# Run with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# Test UI:  http://localhost:8000/docs
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib, json
import numpy as np
import pandas as pd
from compatibility import compute_compatibility_score

app = FastAPI(
    title="Bone Marrow Donor Matching API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load all 3 models
m_survival = joblib.load('model_survival.pkl')
m_relapse  = joblib.load('model_relapse.pkl')
m_gvhd     = joblib.load('model_gvhd.pkl')
encoders   = joblib.load('encoders.pkl')
with open('feature_cols.json') as f:
    feature_cols = json.load(f)

CATEGORICALS = [
    'donor_ABO', 'donor_CMV', 'recipient_ABO', 'recipient_rh',
    'recipient_CMV', 'disease', 'disease_group', 'risk_group',
    'stem_cell_source', 'tx_post_relapse'
]

# Dataset medians for post-transplant fields
ANC_MEDIAN = 16.0
PLT_MEDIAN = 25.0


# ============================================================
# REQUEST SCHEMAS
# — ANC and PLT removed (post-transplant, hardcoded to median)
# — CD3_to_CD34_ratio removed (auto-calculated from CD3/CD34)
# — recipient_age has no upper limit (dataset limitation noted)
# ============================================================

class DonorInput(BaseModel):
    donor_age:         float            # years
    donor_ABO:         str              # '0', 'A', 'B', 'AB'
    donor_CMV:         str              # 'absent', 'present'
    donor_gender:      str              # 'male', 'female'
    antigen:           int              # 0-3, from HLA lab report
    allel:             int              # 0-4, from HLA lab report
    CD34_x1e6_per_kg:  Optional[float] = 10.0  # from lab
    CD3_x1e8_per_kg:   Optional[float] = 5.0   # from lab
    stem_cell_source:  Optional[str]   = 'peripheral_blood'
    # CD3_to_CD34_ratio is auto-calculated from CD3/CD34
    # ANC_recovery and PLT_recovery are post-transplant — hardcoded to median

class PatientInput(BaseModel):
    recipient_age:       float          # years (no upper limit)
    recipient_gender:    str            # 'male', 'female'
    recipient_body_mass: float          # kg
    recipient_ABO:       str            # '0', 'A', 'B', 'AB'
    recipient_rh:        str            # 'plus', 'minus'
    recipient_CMV:       str            # 'absent', 'present'
    disease:             str            # 'ALL','AML','chronic','nonmalignant','lymphoma'
    disease_group:       str            # 'malignant', 'nonmalignant'
    risk_group:          str            # 'high', 'low'
    tx_post_relapse:     Optional[str] = 'no'

class PredictRequest(BaseModel):
    donor:   DonorInput
    patient: PatientInput


# ============================================================
# HELPER: BUILD FEATURE VECTOR
# ============================================================

def build_feature_vector(donor: dict, patient: dict, derived: dict):

    # Auto-calculate CD3 to CD34 ratio
    cd34 = donor.get('CD34_x1e6_per_kg', 10.0)
    cd3  = donor.get('CD3_x1e8_per_kg',  5.0)
    cd3_to_cd34_ratio = round(cd3 / cd34, 6) if cd34 != 0 else 0.0

    row = {
        # Pre-transplant donor fields
        'donor_age':          donor['donor_age'],
        'CD34_x1e6_per_kg':   cd34,
        'CD3_x1e8_per_kg':    cd3,
        'CD3_to_CD34_ratio':  cd3_to_cd34_ratio,   # auto-calculated

        # Pre-transplant patient fields
        'recipient_age':       patient['recipient_age'],
        'recipient_body_mass': patient['recipient_body_mass'],

        # Post-transplant — hardcoded to dataset median
        'ANC_recovery':        ANC_MEDIAN,
        'PLT_recovery':        PLT_MEDIAN,

        # Auto-derived compatibility fields
        'HLA_match_score':    {'10/10':4,'9/10':3,'8/10':2,'7/10':1}.get(
                                  derived['HLA_match'], 1),
        'CMV_status':         derived['CMV_status'],
        'ABO_match_binary':   1 if derived['ABO_match'] == 'matched' else 0,
        'gender_risk':        1 if derived['gender_match'] == 'female_to_male' else 0,
        'donor_age_risk':     1 if donor['donor_age'] >= 35 else 0,
        'total_HLA_diff':     donor['antigen'] + donor['allel'],
        'antigen':            donor['antigen'],
        'allel':              donor['allel'],

        # Categorical fields
        'donor_ABO':          donor['donor_ABO'],
        'donor_CMV':          donor['donor_CMV'],
        'recipient_ABO':      patient['recipient_ABO'],
        'recipient_rh':       patient['recipient_rh'],
        'recipient_CMV':      patient['recipient_CMV'],
        'disease':            patient['disease'],
        'disease_group':      patient['disease_group'],
        'risk_group':         patient['risk_group'],
        'stem_cell_source':   donor['stem_cell_source'],
        'tx_post_relapse':    patient['tx_post_relapse'],
    }

    df_row = pd.DataFrame([row])
    for col in CATEGORICALS:
        le  = encoders[col]
        val = str(df_row[col].iloc[0])
        df_row[col + '_enc'] = le.transform([val]) if val in le.classes_ else [0]

    # Use exact feature order from training
    return df_row[feature_cols].astype(float)


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {"message": "Bone Marrow Matching API is running ✅"}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        donor   = request.donor.dict()
        patient = request.patient.dict()

        # Step 1: Compatibility score + derived fields
        compat  = compute_compatibility_score(donor, patient)
        derived = compat['derived_fields']

        # Step 2: Build feature vector
        X_input = build_feature_vector(donor, patient, derived)

        # Step 3: Run all 3 models
        s_pred = m_survival.predict(X_input)[0]
        s_prob = m_survival.predict_proba(X_input)[0]

        r_pred = m_relapse.predict(X_input)[0]
        r_prob = m_relapse.predict_proba(X_input)[0]

        g_pred = m_gvhd.predict(X_input)[0]
        g_prob = m_gvhd.predict_proba(X_input)[0]

        # Auto-calculated ratio for transparency in response
        cd34 = donor.get('CD34_x1e6_per_kg', 10.0)
        cd3  = donor.get('CD3_x1e8_per_kg',  5.0)
        ratio = round(cd3 / cd34, 6) if cd34 != 0 else 0.0

        # Step 4: Return full result
        return {
            # Compatibility
            "compatibility_score": compat['compatibility_score'],
            "grade":               compat['grade'],
            "breakdown":           compat['breakdown'],
            "derived_fields":      derived,

            # Auto-calculated fields (for frontend display)
            "auto_calculated": {
                "CD3_to_CD34_ratio": ratio,
            },

            # Survival
            "survival_prediction": "Alive" if s_pred == 0 else "At Risk",
            "survival_probability": {
                "alive":   round(float(s_prob[0]) * 100, 1),
                "at_risk": round(float(s_prob[1]) * 100, 1),
            },

            # Relapse
            "relapse_prediction": "Low Risk" if r_pred == 0 else "High Risk",
            "relapse_probability": {
                "low":  round(float(r_prob[0]) * 100, 1),
                "high": round(float(r_prob[1]) * 100, 1),
            },

            # GvHD
            "gvhd_prediction": "Low Risk" if g_pred == 0 else "High Risk",
            "gvhd_probability": {
                "low":  round(float(g_prob[0]) * 100, 1),
                "high": round(float(g_prob[1]) * 100, 1),
            },

            # Verdict
            "recommendation": (
                "✅ Recommended match"
                if compat['compatibility_score'] >= 70 and s_pred == 0
                else "⚠️ Review carefully before proceeding"
            ),

            # Notes for frontend
            "notes": {
                "ANC_PLT": "ANC and PLT recovery are post-transplant measurements, "
                           "set to dataset medians (ANC=16, PLT=25).",
                "age":     "Model trained on pediatric patients (0-20 years). "
                           "Predictions for older patients may be less accurate.",
                "ratio":   f"CD3/CD34 ratio auto-calculated as {ratio}"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))