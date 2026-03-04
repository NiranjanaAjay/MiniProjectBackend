# ============================================================
# test_terminal.py
# Run AFTER pipeline.py
# Command: python test_terminal.py
# ============================================================

import joblib, json
import numpy as np
import pandas as pd
from compatibility import compute_compatibility_score

# ---- Load models ----
print("\nLoading models...", end=" ")
m_survival = joblib.load('model_survival.pkl')
m_relapse  = joblib.load('model_relapse.pkl')
m_gvhd     = joblib.load('model_gvhd.pkl')
encoders   = joblib.load('encoders.pkl')
with open('feature_cols.json') as f:
    feature_cols = json.load(f)
print("Done ✅")

CATEGORICALS = [
    'donor_ABO', 'donor_CMV', 'recipient_ABO', 'recipient_rh',
    'recipient_CMV', 'disease', 'disease_group', 'risk_group',
    'stem_cell_source', 'tx_post_relapse'
]

# Dataset medians for post-transplant fields
ANC_MEDIAN = 16.0
PLT_MEDIAN = 25.0


# ============================================================
# INPUT HELPERS
# ============================================================

def ask_float(prompt, min_val=None, max_val=None):
    while True:
        try:
            val = float(input(f"  {prompt}: ").strip())
            if min_val is not None and val < min_val:
                print(f"    ⚠️  Must be >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"    ⚠️  Must be <= {max_val}")
                continue
            return val
        except ValueError:
            print("    ⚠️  Enter a valid number.")

def ask_float_optional(prompt, default):
    raw = input(f"  {prompt} [default={default}]: ").strip()
    if raw == '':
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"    ⚠️  Invalid, using default {default}")
        return default

def ask_int(prompt, valid_range):
    while True:
        try:
            val = int(input(f"  {prompt} {list(valid_range)}: ").strip())
            if val in valid_range:
                return val
            print(f"    ⚠️  Choose from {list(valid_range)}")
        except ValueError:
            print("    ⚠️  Enter a valid integer.")

def ask_choice(prompt, choices):
    choices_lower = [c.lower() for c in choices]
    while True:
        val = input(f"  {prompt} ({'/'.join(choices)}): ").strip().lower()
        if val in choices_lower:
            return choices[choices_lower.index(val)]
        print(f"    ⚠️  Choose one of: {', '.join(choices)}")


# ============================================================
# COLLECT DONOR DETAILS
# ============================================================

def get_donor_input():
    print("\n" + "─"*55)
    print("  ENTER DONOR DETAILS")
    print("─"*55)

    donor = {}
    donor['donor_age']         = ask_float("Donor Age (years)", 18, 80)
    donor['donor_ABO']         = ask_choice("Donor Blood Group", ['0','A','B','AB'])
    donor['donor_CMV']         = ask_choice("Donor CMV Status", ['absent','present'])
    donor['donor_gender']      = ask_choice("Donor Gender", ['male','female'])

    print("\n  HLA Typing (from lab report):")
    donor['antigen']           = ask_int("  Antigen differences", range(0, 4))
    donor['allel']             = ask_int("  Allel differences",   range(0, 5))

    print("\n  Cell Dose (from lab, press Enter to use default):")
    donor['CD34_x1e6_per_kg']  = ask_float_optional("CD34+ cell dose (x10^6/kg)", 10.0)
    donor['CD3_x1e8_per_kg']   = ask_float_optional("CD3+ cell dose (x10^8/kg)",   5.0)
    
    # Auto-calculated from CD34 and CD3 values
    if donor['CD34_x1e6_per_kg'] != 0:
        donor['CD3_to_CD34_ratio'] = round(donor['CD3_x1e8_per_kg'] / donor['CD34_x1e6_per_kg'], 6)
    else:
        donor['CD3_to_CD34_ratio'] = 0.0
    print(f"  CD3 to CD34 ratio (auto-calculated): {donor['CD3_to_CD34_ratio']}") 
    donor['stem_cell_source']  = ask_choice(
            "Stem Cell Source", ['peripheral_blood','bone_marrow']
        )

    # Post-transplant fields — hardcoded to dataset median
    donor['ANC_recovery'] = ANC_MEDIAN
    donor['PLT_recovery'] = PLT_MEDIAN

    return donor


# ============================================================
# COLLECT PATIENT DETAILS
# ============================================================

def get_patient_input():
    print("\n" + "─"*55)
    print("  ENTER PATIENT DETAILS")
    print("─"*55)

    patient = {}
    patient['recipient_age'] = ask_float("Recipient Age (years)", 0)
    patient['recipient_gender']    = ask_choice("Recipient Gender", ['male','female'])
    patient['recipient_body_mass'] = ask_float("Recipient Body Mass (kg)", 1, 200)
    patient['recipient_ABO']       = ask_choice("Recipient Blood Group", ['0','A','B','AB'])
    patient['recipient_rh']        = ask_choice("Recipient Rh Factor", ['plus','minus'])
    patient['recipient_CMV']       = ask_choice("Recipient CMV Status", ['absent','present'])
    patient['disease']             = ask_choice(
        "Disease Type", ['ALL','AML','chronic','nonmalignant','lymphoma']
    )
    patient['disease_group']       = ask_choice(
        "Disease Group", ['malignant','nonmalignant']
    )
    patient['risk_group']          = ask_choice("Risk Group", ['high','low'])
    patient['tx_post_relapse']     = ask_choice("Post-relapse transplant", ['no','yes'])

    return patient


# ============================================================
# PREDICT
# ============================================================

def predict(donor, patient):
    compat  = compute_compatibility_score(donor, patient)
    derived = compat['derived_fields']

    # Build row with ALL fields the model was trained on
    row = {
        # Pre-transplant donor fields
        'donor_age':           donor['donor_age'],
        'CD34_x1e6_per_kg':    donor['CD34_x1e6_per_kg'],
        'CD3_x1e8_per_kg':     donor['CD3_x1e8_per_kg'],
        'CD3_to_CD34_ratio':   donor['CD3_to_CD34_ratio'],

        # Pre-transplant patient fields
        'recipient_age':       patient['recipient_age'],
        'recipient_body_mass': patient['recipient_body_mass'],

        # Post-transplant — hardcoded to dataset median
        'ANC_recovery':        ANC_MEDIAN,
        'PLT_recovery':        PLT_MEDIAN,

        # Auto-derived compatibility fields
        'HLA_match_score':     {'10/10':4,'9/10':3,'8/10':2,'7/10':1}.get(
                                   derived['HLA_match'], 1),
        'CMV_status':          derived['CMV_status'],
        'ABO_match_binary':    1 if derived['ABO_match'] == 'matched' else 0,
        'gender_risk':         1 if derived['gender_match'] == 'female_to_male' else 0,
        'donor_age_risk':      1 if donor['donor_age'] >= 35 else 0,
        'total_HLA_diff':      donor['antigen'] + donor['allel'],
        'antigen':             donor['antigen'],
        'allel':               donor['allel'],

        # Categorical fields
        'donor_ABO':           donor['donor_ABO'],
        'donor_CMV':           donor['donor_CMV'],
        'recipient_ABO':       patient['recipient_ABO'],
        'recipient_rh':        patient['recipient_rh'],
        'recipient_CMV':       patient['recipient_CMV'],
        'disease':             patient['disease'],
        'disease_group':       patient['disease_group'],
        'risk_group':          patient['risk_group'],
        'stem_cell_source':    donor['stem_cell_source'],
        'tx_post_relapse':     patient['tx_post_relapse'],
    }

    df_row = pd.DataFrame([row])

    # Encode categoricals
    for col in CATEGORICALS:
        le  = encoders[col]
        val = str(df_row[col].iloc[0])
        df_row[col + '_enc'] = le.transform([val]) if val in le.classes_ else [0]

    # Use EXACT feature order from training — this fixes the error
    X_input = df_row[feature_cols].astype(float)

    s_pred = m_survival.predict(X_input)[0]
    s_prob = m_survival.predict_proba(X_input)[0]
    r_pred = m_relapse.predict(X_input)[0]
    r_prob = m_relapse.predict_proba(X_input)[0]
    g_pred = m_gvhd.predict(X_input)[0]
    g_prob = m_gvhd.predict_proba(X_input)[0]

    return compat, derived, s_pred, s_prob, r_pred, r_prob, g_pred, g_prob


# ============================================================
# PRINT REPORT
# ============================================================

def print_report(compat, derived, s_pred, s_prob, r_pred, r_prob, g_pred, g_prob):
    score = compat['compatibility_score']
    grade = compat['grade']
    grade_icon = {'Excellent':'🌟','Good':'✅','Moderate':'🟡','Poor':'🔴'}.get(grade,'')

    print("\n\n" + "="*55)
    print("     BONE MARROW TRANSPLANT MATCHING REPORT")
    print("="*55)

    print(f"\n  COMPATIBILITY SCORE : {score}/100  —  {grade} {grade_icon}")

    # Score breakdown table
    print(f"\n  {'Factor':<22} {'Score':>5}   {'Bar':<12}  Notes")
    print(f"  {'─'*22}  {'─'*5}   {'─'*12}  {'─'*22}")
    for factor, detail in compat['breakdown'].items():
        filled = int((detail['score'] / detail['max']) * 12)
        bar    = '█' * filled + '░' * (12 - filled)
        print(f"  {factor:<22} {detail['score']:>2}/{detail['max']:<2}"
              f"   {bar}  {detail['label']}")

    # Auto-derived fields
    print(f"\n  Auto-computed fields:")
    print(f"    HLA Match     : {derived['HLA_match']}")
    print(f"    HLA Mismatch  : {derived['HLA_mismatch']}")
    print(f"    ABO Match     : {derived['ABO_match']}")
    print(f"    CMV Status    : {derived['CMV_status']}  (0=best, 3=worst)")
    print(f"    Gender Match  : {derived['gender_match']}")

    # Predictions
    print(f"\n  {'─'*55}")
    print(f"  PREDICTIONS")
    print(f"  {'─'*55}")

    def bar20(prob):
        return '█' * int(prob * 20)

    s_icon = '✅' if s_pred == 0 else '⚠️ '
    r_icon = '✅' if r_pred == 0 else '⚠️ '
    g_icon = '✅' if g_pred == 0 else '⚠️ '

    print(f"\n  SURVIVAL PROBABILITY     {s_icon} {'Alive' if s_pred==0 else 'At Risk'}")
    print(f"    Alive    {s_prob[0]*100:5.1f}%  {bar20(s_prob[0])}")
    print(f"    At Risk  {s_prob[1]*100:5.1f}%  {bar20(s_prob[1])}")

    print(f"\n  RELAPSE RISK             {r_icon} {'Low Risk' if r_pred==0 else 'High Risk'}")
    print(f"    Low      {r_prob[0]*100:5.1f}%  {bar20(r_prob[0])}")
    print(f"    High     {r_prob[1]*100:5.1f}%  {bar20(r_prob[1])}")

    print(f"\n  GvHD RISK (Stage III/IV) {g_icon} {'Low Risk' if g_pred==0 else 'High Risk'}")
    print(f"    Low      {g_prob[0]*100:5.1f}%  {bar20(g_prob[0])}")
    print(f"    High     {g_prob[1]*100:5.1f}%  {bar20(g_prob[1])}")

    # Verdict
    print(f"\n  {'='*55}")
    if score >= 70 and s_pred == 0:
        print("  VERDICT  :  ✅  RECOMMENDED MATCH")
    elif score >= 50:
        print("  VERDICT  :  🟡  POSSIBLE MATCH — REVIEW CAREFULLY")
    else:
        print("  VERDICT  :  ⚠️   NOT RECOMMENDED — SEEK BETTER DONOR")
    print(f"  {'='*55}")

# ============================================================
# MAIN LOOP
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*55)
    print("     BONE MARROW TRANSPLANT MATCHING SYSTEM")
    print("="*55)
    print("  Enter donor and patient details below.")
    print("  Press Ctrl+C at any time to exit.")

    while True:
        try:
            donor   = get_donor_input()
            patient = get_patient_input()

            print("\n  Calculating...", end=" ", flush=True)
            results = predict(donor, patient)
            print("Done ✅")

            print_report(*results)

            again = input("  Run another prediction? (yes/no): ").strip().lower()
            if again not in ('yes', 'y'):
                print("\n  Exiting. Goodbye! 👋\n")
                break

        except KeyboardInterrupt:
            print("\n\n  Exiting. Goodbye! 👋\n")
            break