# Computes antigen and allel differences from raw HLA profiles

ANTIGEN_LOCI = ['HLA-A', 'HLA-B', 'HLA-C']
ALLEL_LOCI   = ['HLA-DRB1', 'HLA-DQB1']

def extract_hla_profile(row: dict, prefix: str) -> dict:
    """
    Extracts HLA profile from a dataframe row.
    prefix is 'donor' or 'patient' (matches column names in CSV)
    """
    # Column names like HLA_A_1, HLA_A_2, HLA_B_1 etc.
    return {
        'HLA-A':    [row[f'HLA_A_1'],    row[f'HLA_A_2']],
        'HLA-B':    [row[f'HLA_B_1'],    row[f'HLA_B_2']],
        'HLA-C':    [row[f'HLA_C_1'],    row[f'HLA_C_2']],
        'HLA-DRB1': [row[f'HLA_DRB1_1'], row[f'HLA_DRB1_2']],
        'HLA-DQB1': [row[f'HLA_DQB1_1'], row[f'HLA_DQB1_2']],
    }

def compute_hla_differences(donor_hla: dict, patient_hla: dict):
    """
    Returns (antigen_diff, allel_diff)
    antigen_diff: mismatches in HLA-A, B, C (0-3)
    allel_diff:   mismatches in HLA-DRB1, DQB1 (0-2)
    """
    antigen_diff = sum(
        1 for locus in ANTIGEN_LOCI
        if not set(donor_hla[locus]).intersection(set(patient_hla[locus]))
    )
    allel_diff = sum(
        1 for locus in ALLEL_LOCI
        if not set(donor_hla[locus]).intersection(set(patient_hla[locus]))
    )
    return antigen_diff, allel_diff

def get_hla_match_label(antigen_diff, allel_diff):
    total = antigen_diff + allel_diff
    return {0:'10/10', 1:'9/10', 2:'8/10'}.get(min(total, 2), '7/10')