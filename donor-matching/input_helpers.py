# ============================================================
# input_helpers.py
# All terminal input functions with validation
# ============================================================

HLA_ALLELES = {
    'HLA-A':    ['01:01','02:01','03:01','24:02','11:01',
                 '29:02','31:01','32:01','68:01','23:01'],
    'HLA-B':    ['07:02','08:01','15:01','44:02','35:01',
                 '51:01','40:01','18:01','57:01','44:03'],
    'HLA-C':    ['07:01','07:02','04:01','03:04','06:02',
                 '05:01','12:03','02:02','08:02','03:03'],
    'HLA-DRB1': ['15:01','07:01','03:01','13:01','04:01',
                 '01:01','11:01','04:04','08:01','12:01'],
    'HLA-DQB1': ['06:02','02:01','03:01','05:01','03:02',
                 '06:03','04:02','02:02','05:02','06:01'],
}


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


def ask_hla_profile(label):
    """Ask user to select HLA alleles for each locus from numbered list"""
    print(f"\n  {label} HLA Profile (select alleles):")
    profile = {}
    for locus, alleles in HLA_ALLELES.items():
        print(f"\n    {locus} options:")
        for i, a in enumerate(alleles, 1):
            print(f"      {i:>2}. {a}")
        while True:
            try:
                a1 = int(input(f"    {locus} Allele 1 (1-{len(alleles)}): ").strip())
                a2 = int(input(f"    {locus} Allele 2 (1-{len(alleles)}): ").strip())
                if 1 <= a1 <= len(alleles) and 1 <= a2 <= len(alleles):
                    profile[locus] = [alleles[a1-1], alleles[a2-1]]
                    break
                print(f"    ⚠️  Enter numbers between 1 and {len(alleles)}")
            except ValueError:
                print("    ⚠️  Enter a valid number.")
    return profile


def get_donor_input():
    print("\n  ENTER DONOR DETAILS")
    print("  " + "─"*50)
    donor = {}
    donor['donor_age']        = ask_float("Donor Age (years)", 18, 80)
    donor['donor_ABO']        = ask_choice("Donor Blood Group", ['0','A','B','AB'])
    donor['donor_CMV']        = ask_choice("Donor CMV Status", ['absent','present'])
    donor['donor_gender']     = ask_choice("Donor Gender", ['male','female'])
    print("\n  Cell Dose (press Enter for default):")
    donor['CD34_x1e6_per_kg']  = ask_float_optional("CD34+ cell dose (x10^6/kg)", 10.0)
    donor['CD3_x1e8_per_kg']   = ask_float_optional("CD3+ cell dose (x10^8/kg)",   5.0)
    donor['CD3_to_CD34_ratio'] = round(
        donor['CD3_x1e8_per_kg'] / donor['CD34_x1e6_per_kg']
        if donor['CD34_x1e6_per_kg'] != 0 else 0.0, 6
    )
    print(f"  CD3/CD34 ratio (auto): {donor['CD3_to_CD34_ratio']}")
    donor['stem_cell_source'] = ask_choice(
        "Stem Cell Source", ['peripheral_blood','bone_marrow']
    )
    return donor


def get_patient_input():
    print("\n  ENTER PATIENT DETAILS")
    print("  " + "─"*50)
    patient = {}
    patient['recipient_age']       = ask_float("Recipient Age (years)", 0)
    patient['recipient_gender']    = ask_choice("Recipient Gender", ['male','female'])
    patient['recipient_body_mass'] = ask_float("Recipient Body Mass (kg)", 1, 200)
    patient['recipient_ABO']       = ask_choice("Recipient Blood Group",
                                                 ['0','A','B','AB'])
    patient['recipient_rh']        = ask_choice("Recipient Rh Factor", ['plus','minus'])
    patient['recipient_CMV']       = ask_choice("Recipient CMV Status",
                                                 ['absent','present'])
    patient['disease']             = ask_choice(
        "Disease Type", ['ALL','AML','chronic','nonmalignant','lymphoma']
    )
    patient['disease_group']       = ask_choice(
        "Disease Group", ['malignant','nonmalignant']
    )
    patient['risk_group']          = ask_choice("Risk Group", ['high','low'])
    patient['tx_post_relapse']     = ask_choice("Post-relapse transplant", ['no','yes'])
    return patient