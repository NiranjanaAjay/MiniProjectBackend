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
from dotenv import load_dotenv
import os

load_dotenv()

from option_a import run_option_a
import requests
import builtins

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

SUPABASE_URL, SUPABASE_KEY = (
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY"),
)

supabase_headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal"
}

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


@app.post("/find-top5/{patient_id}")
def find_top5(patient_id: int):

    # Fetch patient
    patient_res = requests.get(
        "https://uhpinfogzptzsvulhpvr.supabase.co/rest/v1/Patient",
        headers=supabase_headers,
        params={"Patient_id": f"eq.{patient_id}"}  # capital P
    )

    patient_data = patient_res.json()
    print("Patient data:", patient_data)
    
    if not patient_data:
        raise HTTPException(status_code=404, detail="Patient not found")

    patient = patient_data[0]

    # Rename + map patient fields to match with upa table
    patient = {
        "patient_id": patient["Patient_id"],

        "recipient_age": float(patient["Age"]),
        "recipient_gender": patient["Gender"].lower(),
        "recipient_body_mass": float(patient["BodyMass"]),
        "recipient_ABO": patient["BloodGroup"].replace("O", "0"),
        "recipient_rh": {
            "+": "plus", "-": "minus",
            "positive": "plus", "negative": "minus",
            "plus": "plus", "minus": "minus"          # ← add these
        }.get(patient["RhFactor"].lower(), "plus"),

        "recipient_CMV": {
            "Positive": "present", "Negative": "absent",
            "Present": "present",  "Absent": "absent"  # ← add these
        }.get(patient["CMVStatus"], "absent"),
        "disease": {
            "Leukemia": "AML",
            "ALL": "ALL",
            "Lymphoma": "lymphoma"
        }.get(patient["DiseaseType"], "AML"),
        "disease_group": patient["DiseaseGroup"],
        "risk_group": patient["RiskGroup"],
        "tx_post_relapse": patient["PostRelapse"],

        # HLA fields 
        "HLA_A_1": patient["Hla_a_1"],
        "HLA_A_2": patient["Hla_a_2"],
        "HLA_B_1": patient["Hla_b_1"],
        "HLA_B_2": patient["Hla_b_2"],
        "HLA_C_1": patient["Hla_c_1"],
        "HLA_C_2": patient["Hla_c_2"],
        "HLA_DRB1_1": patient["Hla_drb1_1"],
        "HLA_DRB1_2": patient["Hla_drb1_2"],
        "HLA_DQB1_1": patient["Hla_dqb1_1"],
        "HLA_DQB1_2": patient["Hla_dqb1_2"],
    }

    # Fetch all donors
    donor_res = requests.get(
        "https://uhpinfogzptzsvulhpvr.supabase.co/rest/v1/Donor",
        headers=supabase_headers
    )

    donors = donor_res.json()

    if not donors:
        raise HTTPException(status_code=404, detail="No donors found")

    donor_df = pd.DataFrame(donors)

    # Rename columns for donor to match with table
    donor_df = donor_df.rename(columns={
        "Donor_id": "donor_id",
        "Age": "donor_age",
        "BloodGroup": "donor_ABO",
        "CMVStatus": "donor_CMV",
        "Gender": "donor_gender",
        "StemCellSource": "stem_cell_source",

        "Hla_a_1": "HLA_A_1",
        "Hla_a_2": "HLA_A_2",
        "Hla_b_1": "HLA_B_1",
        "Hla_b_2": "HLA_B_2",
        "Hla_c_1": "HLA_C_1",
        "Hla_c_2": "HLA_C_2",
        "Hla_drb1_1": "HLA_DRB1_1",
        "Hla_drb1_2": "HLA_DRB1_2",
        "Hla_dqb1_1": "HLA_DQB1_1",
        "Hla_dqb1_2": "HLA_DQB1_2",
    })

    # Add missing fields
    donor_df["CD34_x1e6_per_kg"] = 10.0
    donor_df["CD3_x1e8_per_kg"] = 5.0
    donor_df["CD3_to_CD34_ratio"] = donor_df["CD3_x1e8_per_kg"] / donor_df["CD34_x1e6_per_kg"] 

    # Fix formats
    donor_df["donor_ABO"] = donor_df["donor_ABO"].replace({"O": "0"})
    donor_df["donor_CMV"] = donor_df["donor_CMV"].map({
        "Positive": "present",
        "Negative": "absent"
    })

    # Run ML
    top5 = run_option_a(patient, donor_df)

    # Prepare insert data
    insert_data = []

    for rank, d in enumerate(top5, 1):
        insert_data.append({
            "Patient_id": int(patient["patient_id"]),
            "Patient_Name": f"Patient {patient['patient_id']}",

            "Donor_id": d["donor_id"],
            "Donor_Name": f"Donor {d['donor_id']}",

            "CompatabilityScore": d["compatibility_score"],
            "HlaMatch": d["hla_match"],
            "AboMatch": d["abo_match"],

            "Antigen":d["antigen_diff"],
            "Allele":d["allel_diff"],

            "Survival": d["alive_probability"],
            "RelapseRisk": d["relapse_risk"],
            "GvhdRisk": d["gvhd_risk"]
        })

    # Delete existing matches for this patient before inserting new ones
    requests.delete(
        SUPABASE_URL,
        headers=supabase_headers,
        params={"Patient_id": f"eq.{patient['patient_id']}"}
    )

    #  Insert into Supabase
    res = requests.post(
        SUPABASE_URL,
        json=insert_data,
        headers=supabase_headers
    )

    if res.status_code >= 300:
        raise HTTPException(status_code=500, detail=res.text)

    return {
        "message": "Top 5 generated and stored ✅",
        "data": insert_data
    }