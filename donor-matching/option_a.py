# ============================================================
# option_a.py
# Option A — Select patient → score all donors → show top 5
# ============================================================

import pandas as pd
from compute_hla import extract_hla_profile, compute_hla_differences
from predict_core import build_and_predict

# Load databases
donor_db   = pd.read_csv('donor_db.csv')
patient_db = pd.read_csv('patient_db.csv')


# def select_patient_from_db():
#     print("\n" + "─"*60)
#     print("  AVAILABLE PATIENTS")
#     print("─"*60)
#     print(f"  {'ID':<6} {'Age':<6} {'Gender':<8} {'Blood':<7}"
#           f" {'Disease':<14} {'Risk':<6}")
#     print(f"  {'─'*6} {'─'*6} {'─'*8} {'─'*7} {'─'*14} {'─'*6}")

#     for _, row in patient_db.iterrows():
#         print(f"  {int(row['patient_id']):<6} "
#               f"{row['recipient_age']:<6} "
#               f"{row['recipient_gender']:<8} "
#               f"{row['recipient_ABO']:<7} "
#               f"{row['disease']:<14} "
#               f"{row['risk_group']:<6}")

#     while True:
#         try:
#             pid   = int(input("\n  Enter Patient ID: ").strip())
#             match = patient_db[patient_db['patient_id'] == pid]
#             if len(match) == 0:
#                 print(f"    ⚠️  Patient ID {pid} not found.")
#                 continue
#             return match.iloc[0].to_dict()
#         except ValueError:
#             print("    ⚠️  Enter a valid number.")




def score_donor_against_patient(donor_row: dict, patient_row: dict):
    donor_hla   = extract_hla_profile(donor_row,   'donor')
    patient_hla = extract_hla_profile(patient_row, 'patient')
    antigen_diff, allel_diff = compute_hla_differences(donor_hla, patient_hla)

    donor = {
        'donor_age':         donor_row['donor_age'],
        'donor_ABO':         donor_row['donor_ABO'],
        'donor_CMV':         donor_row['donor_CMV'],
        'donor_gender':      donor_row['donor_gender'],
        'antigen':           antigen_diff,
        'allel':             allel_diff,
        'CD34_x1e6_per_kg':  donor_row['CD34_x1e6_per_kg'],
        'CD3_x1e8_per_kg':   donor_row['CD3_x1e8_per_kg'],
        'CD3_to_CD34_ratio': donor_row['CD3_to_CD34_ratio'],
        'stem_cell_source':  donor_row['stem_cell_source'],
    }
    patient = {
        'recipient_age':       patient_row['recipient_age'],
        'recipient_gender':    patient_row['recipient_gender'],
        'recipient_body_mass': patient_row['recipient_body_mass'],
        'recipient_ABO':       patient_row['recipient_ABO'],
        'recipient_rh':        patient_row['recipient_rh'],
        'recipient_CMV':       patient_row['recipient_CMV'],
        'disease':             patient_row['disease'],
        'disease_group':       patient_row['disease_group'],
        'risk_group':          patient_row['risk_group'],
        'tx_post_relapse':     patient_row['tx_post_relapse'],
    }

    compat, derived, s_pred, s_prob, r_pred, r_prob, g_pred, g_prob = \
        build_and_predict(donor, patient, antigen_diff, allel_diff)

    return {
        'donor_id':            int(donor_row['donor_id']),
        'donor_age':           donor['donor_age'],
        'donor_ABO':           donor['donor_ABO'],
        'donor_CMV':           donor['donor_CMV'],
        'donor_gender':        donor['donor_gender'],
        'stem_cell_source':    donor['stem_cell_source'],
        'antigen_diff':        antigen_diff,
        'allel_diff':          allel_diff,
        'hla_match':           derived['HLA_match'],
        'abo_match':           derived['ABO_match'],
        'compatibility_score': compat['compatibility_score'],
        'grade':               compat['grade'],
        'alive_probability':   round(float(s_prob[0]) * 100, 1),
        'relapse_risk':        round(float(r_prob[1]) * 100, 1),
        'gvhd_risk':           round(float(g_prob[1]) * 100, 1),
    }


# def print_top5_report(top5: list, patient: dict):
    print("\n\n" + "="*60)
    print("           TOP 5 DONOR MATCHES")
    print("="*60)
    print(f"  Patient ID  : {int(patient['patient_id'])}")
    print(f"  Age         : {patient['recipient_age']}")
    print(f"  Blood Group : {patient['recipient_ABO']}")
    print(f"  Disease     : {patient['disease']}")
    print(f"  Risk Group  : {patient['risk_group']}")
    print(f"  HLA-A       : {patient['HLA_A_1']} / {patient['HLA_A_2']}")
    print(f"  HLA-B       : {patient['HLA_B_1']} / {patient['HLA_B_2']}")
    print(f"  HLA-C       : {patient['HLA_C_1']} / {patient['HLA_C_2']}")
    print(f"  HLA-DRB1    : {patient['HLA_DRB1_1']} / {patient['HLA_DRB1_2']}")
    print(f"  HLA-DQB1    : {patient['HLA_DQB1_1']} / {patient['HLA_DQB1_2']}")
    print("="*60)

    for rank, d in enumerate(top5, 1):
        grade_icon = {
            'Excellent':'🌟','Good':'✅','Moderate':'🟡','Poor':'🔴'
        }.get(d['grade'], '')
        print(f"\n  RANK #{rank}  —  Donor ID: {d['donor_id']}  "
              f"({d['grade']} {grade_icon})")
        print(f"  {'─'*55}")
        print(f"  Compatibility   : {d['compatibility_score']}/100")
        print(f"  HLA Match       : {d['hla_match']}  "
              f"(antigen={d['antigen_diff']}, allel={d['allel_diff']})")
        print(f"  ABO Match       : {d['abo_match']}")
        print(f"  Donor Age       : {d['donor_age']}  "
              f"({'Optimal ✅' if d['donor_age'] < 35 else 'Suboptimal'})")
        print(f"  Donor Blood     : {d['donor_ABO']}")
        print(f"  Donor CMV       : {d['donor_CMV']}")
        print(f"  Donor Gender    : {d['donor_gender']}")
        print(f"  Stem Cell       : {d['stem_cell_source']}")
        print(f"  {'─'*55}")
        print(f"  Survival(Alive) : {d['alive_probability']}%  "
              f"{'█' * int(d['alive_probability']/5)}")
        print(f"  Relapse Risk    : {d['relapse_risk']}%  "
              f"{'█' * int(d['relapse_risk']/5)}")
        print(f"  GvHD Risk       : {d['gvhd_risk']}%  "
              f"{'█' * int(d['gvhd_risk']/5)}")

    print(f"\n{'='*60}\n")


def run_option_a(patient: dict, donor_db: pd.DataFrame):

    results = []

    for _, donor_row in donor_db.iterrows():
        try:
            result = score_donor_against_patient(
                donor_row.to_dict(),
                patient
            )
            results.append(result)
        except Exception:
            continue

    results.sort(
        key=lambda x: (x['compatibility_score'], x['alive_probability']),
        reverse=True
    )

    top5 = results[:5]

    return top5