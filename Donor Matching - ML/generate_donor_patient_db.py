# Generates donor_db.csv and patient_db.csv with raw HLA profiles

import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

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

def random_hla():
    return {
        locus: random.sample(alleles, 2)
        for locus, alleles in HLA_ALLELES.items()
    }

# ---- Generate donor database ----
def generate_donors(n=100):
    rows = []
    for i in range(n):
        hla = random_hla()
        CD34 = round(random.uniform(1.0, 30.0), 2)
        CD3  = round(random.uniform(0.5, 15.0), 2)
        rows.append({
            'donor_id':          i + 1,
            'donor_age':         random.randint(18, 60),
            'donor_ABO':         random.choices(['0','A','B','AB'],
                                     weights=[44,42,10,4])[0],
            'donor_CMV':         random.choices(['absent','present'],
                                     weights=[60,40])[0],
            'donor_gender':      random.choice(['male','female']),
            'CD34_x1e6_per_kg':  CD34,
            'CD3_x1e8_per_kg':   CD3,
            'CD3_to_CD34_ratio': round(CD3/CD34, 6),
            'stem_cell_source':  random.choices(
                                     ['peripheral_blood','bone_marrow'],
                                     weights=[75,25])[0],
            # HLA allele 1 and 2 for each locus
            'HLA_A_1':    hla['HLA-A'][0],
            'HLA_A_2':    hla['HLA-A'][1],
            'HLA_B_1':    hla['HLA-B'][0],
            'HLA_B_2':    hla['HLA-B'][1],
            'HLA_C_1':    hla['HLA-C'][0],
            'HLA_C_2':    hla['HLA-C'][1],
            'HLA_DRB1_1': hla['HLA-DRB1'][0],
            'HLA_DRB1_2': hla['HLA-DRB1'][1],
            'HLA_DQB1_1': hla['HLA-DQB1'][0],
            'HLA_DQB1_2': hla['HLA-DQB1'][1],
        })
    df = pd.DataFrame(rows)
    df.to_csv('donor_db.csv', index=False)
    print(f"Generated {n} donors → donor_db.csv")
    return df

# ---- Generate patient database ----
def generate_patients(n=50):
    rows = []
    for i in range(n):
        hla = random_hla()
        rows.append({
            'patient_id':          i + 1,
            'recipient_age':       round(random.uniform(0.5, 20), 1),
            'recipient_gender':    random.choice(['male','female']),
            'recipient_body_mass': round(random.uniform(5, 80), 1),
            'recipient_ABO':       random.choices(['0','A','B','AB'],
                                       weights=[44,42,10,4])[0],
            'recipient_rh':        random.choices(['plus','minus'],
                                       weights=[85,15])[0],
            'recipient_CMV':       random.choices(['absent','present'],
                                       weights=[55,45])[0],
            'disease':             random.choices(
                                       ['ALL','AML','chronic',
                                        'nonmalignant','lymphoma'],
                                       weights=[40,20,15,15,10])[0],
            'disease_group':       random.choices(['malignant','nonmalignant'],
                                       weights=[85,15])[0],
            'risk_group':          random.choices(['high','low'],
                                       weights=[40,60])[0],
            'tx_post_relapse':     random.choices(['no','yes'],
                                       weights=[85,15])[0],
            # HLA profile
            'HLA_A_1':    hla['HLA-A'][0],
            'HLA_A_2':    hla['HLA-A'][1],
            'HLA_B_1':    hla['HLA-B'][0],
            'HLA_B_2':    hla['HLA-B'][1],
            'HLA_C_1':    hla['HLA-C'][0],
            'HLA_C_2':    hla['HLA-C'][1],
            'HLA_DRB1_1': hla['HLA-DRB1'][0],
            'HLA_DRB1_2': hla['HLA-DRB1'][1],
            'HLA_DQB1_1': hla['HLA-DQB1'][0],
            'HLA_DQB1_2': hla['HLA-DQB1'][1],
        })
    df = pd.DataFrame(rows)
    df.to_csv('patient_db.csv', index=False)
    print(f"Generated {n} patients → patient_db.csv")
    return df

if __name__ == '__main__':
    generate_donors(100)
    generate_patients(50)