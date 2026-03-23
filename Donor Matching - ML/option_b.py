# ============================================================
# option_b.py
# Option B — Enter donor + patient manually → get full report
# Pairwise
# ============================================================

from input_helpers import (get_donor_input, get_patient_input, ask_hla_profile)
from compute_hla import compute_hla_differences
from predict_core import build_and_predict, print_report


def run_option_b():
    print("\n" + "─"*55)
    print("  OPTION B — MANUAL DONOR + PATIENT ENTRY")
    print("─"*55)

    # Donor details
    donor     = get_donor_input()
    donor_hla = ask_hla_profile("Donor")

    # Patient details
    patient     = get_patient_input()
    patient_hla = ask_hla_profile("Patient")

    # Compute HLA differences from profiles
    antigen_diff, allel_diff = compute_hla_differences(donor_hla, patient_hla)
    donor['antigen'] = antigen_diff
    donor['allel']   = allel_diff

    print(f"\n  HLA differences (auto-computed):")
    print(f"    Antigen diff : {antigen_diff}")
    print(f"    Allel diff   : {allel_diff}")

    print("\n  Calculating...", end=" ", flush=True)
    results = build_and_predict(donor, patient, antigen_diff, allel_diff)
    print("Done ✅")

    print_report(*results)