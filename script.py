import pandas as pd
import requests

CSV_PATH     = "/home/kirubha/Projects/MiniProjectBackend/donor-matching/donor_db.csv"
SUPABASE_URL = "https://uhpinfogzptzsvulhpvr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVocGluZm9nenB0enN2dWxocHZyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQyMjQyNjEsImV4cCI6MjA2OTgwMDI2MX0.PrVCuwG314G4x3YW-b3p1-xHDLjcLyLbxvh4fMt_UvE"
REST_URL     = f"{SUPABASE_URL}/rest/v1/Donor"

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal"
}

# CSV column → Supabase table column
COLUMN_MAP = {
    "donor_age":        "Age",
    "donor_ABO":        "BloodGroup",
    "donor_CMV":        "CMVStatus",
    "donor_gender":     "Gender",
    "stem_cell_source": "StemCellSource",
    "HLA_A_1":          "Hla_a_1",
    "HLA_A_2":          "Hla_a_2",
    "HLA_B_1":          "Hla_b_1",
    "HLA_B_2":          "Hla_b_2",
    "HLA_C_1":          "Hla_c_1",
    "HLA_C_2":          "Hla_c_2",
    "HLA_DRB1_1":       "Hla_drb1_1",
    "HLA_DRB1_2":       "Hla_drb1_2",
    "HLA_DQB1_1":       "Hla_dqb1_1",
    "HLA_DQB1_2":       "Hla_dqb1_2",
}

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows")

# Rename CSV cols to match table
df = df.rename(columns=COLUMN_MAP)

# Keep only columns that exist in the table
TABLE_COLS = list(COLUMN_MAP.values())
df = df[[c for c in TABLE_COLS if c in df.columns]]

# Fix formats
df["CMVStatus"] = df["CMVStatus"].map({"present": "Present", "absent": "Absent"})
df["Gender"]    = df["Gender"].str.capitalize()   # male → Male, female → Female

# 1️⃣ Add 100 unique donor names
for i in range(len(df)):
    df.loc[i, "Name"] = f"Donor_{i+1}"

print(f"Inserting {len(df)} records with columns: {df.columns.tolist()}\n")

records = df.where(pd.notna(df), None).to_dict(orient="records")

BATCH_SIZE = 50
for i in range(0, len(records), BATCH_SIZE):
    batch = records[i : i + BATCH_SIZE]
    print(f"Inserting rows {i}–{i+len(batch)}...", end=" ")
    res = requests.post(REST_URL, json=batch, headers=headers, timeout=30)
    print(f"status={res.status_code}")
    if res.status_code >= 300:
        print("❌ Error:", res.text)
        break
    else:
        print("✅ OK")

print("\nDone!")