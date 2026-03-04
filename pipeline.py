# ============================================================
# Run this ONCE to train and save all 3 models
# ============================================================

import pandas as pd
import numpy as np
from scipy.io import arff
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# STEP 1: LOAD ARFF
# ============================================================

def load_arff(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        data, meta = arff.loadarff(f)
    df = pd.DataFrame(data)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(
            lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
        )
    print(f"[1] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    df.to_csv('bone_marrow.csv', index=False)
    print("[1] Saved as bone_marrow.csv")
    return df


# ============================================================
# STEP 2: CLEAN
# ============================================================

def clean_data(df):
    df.replace('?', np.nan, inplace=True)
    numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    missing = df.isnull().sum()
    print(f"\n[2] Missing values found:\n{missing[missing > 0]}")

    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    print(f"[2] Cleaning done. Remaining nulls: {df.isnull().sum().sum()}")
    return df


# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    df['ABO_match_binary'] = (df['ABO_match'] == 'matched').astype(int)

    hla_map = {'10/10': 4, '9/10': 3, '8/10': 2, '7/10': 1}
    df['HLA_match_score'] = df['HLA_match'].map(hla_map).fillna(0)

    df['CMV_status']    = pd.to_numeric(df['CMV_status'], errors='coerce').fillna(0)
    df['gender_risk']   = (df['gender_match'] == 'female_to_male').astype(int)
    df['donor_age_risk']= (df['donor_age'] >= 35).astype(int)
    df['antigen']       = pd.to_numeric(df['antigen'], errors='coerce').fillna(0)
    df['allel']         = pd.to_numeric(df['allel'],   errors='coerce').fillna(0)
    df['total_HLA_diff']= df['antigen'] + df['allel']

    print("[3] Feature engineering done.")
    return df


# ============================================================
# STEP 4: AUGMENTATION
# ============================================================

SAFE_NUMERIC = [
    'donor_age', 'recipient_age', 'recipient_body_mass',
    'CD34_x1e6_per_kg', 'CD3_x1e8_per_kg', 'CD3_to_CD34_ratio',
    'HLA_match_score', 'CMV_status', 'ABO_match_binary',
    'gender_risk', 'donor_age_risk', 'total_HLA_diff',
    'antigen', 'allel'
]

CATEGORICALS = [
    'donor_ABO', 'donor_CMV', 'recipient_ABO', 'recipient_rh',
    'recipient_CMV', 'disease', 'disease_group', 'risk_group',
    'stem_cell_source', 'tx_post_relapse'
]


def augment_data(df, target_rows=450):
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import LabelEncoder

    print(f"\n[4] Original size: {len(df)} rows")
    print(f"[4] Survival distribution:\n{df['survival_status'].value_counts()}")

    SAFE_NUMERIC = [
        'donor_age', 'recipient_age', 'recipient_body_mass',
        'CD34_x1e6_per_kg', 'CD3_x1e8_per_kg', 'CD3_to_CD34_ratio',
        'ANC_recovery', 'PLT_recovery', 'HLA_match_score',
        'CMV_status', 'ABO_match_binary', 'gender_risk',
        'donor_age_risk', 'total_HLA_diff', 'antigen', 'allel'
    ]

    CATEGORICALS = [
        'donor_ABO', 'donor_CMV', 'recipient_ABO', 'recipient_rh',
        'recipient_CMV', 'disease', 'disease_group', 'risk_group',
        'stem_cell_source', 'tx_post_relapse'
    ]

    df_aug = df.copy()
    encoders = {}

    for col in CATEGORICALS:
        le = LabelEncoder()
        df_aug[col + '_enc'] = le.fit_transform(df_aug[col].astype(str))
        encoders[col] = le

    encoded_cats = [c + '_enc' for c in CATEGORICALS]
    feature_cols = SAFE_NUMERIC + encoded_cats

    X = df_aug[feature_cols].astype(float)

    # All 3 targets from ORIGINAL df (same length as X)
    y_survival = df_aug['survival_status'].astype(int)
    y_relapse  = df_aug['relapse'].map({'yes': 1, 'no': 0}).astype(int)
    y_gvhd     = df_aug['acute_GvHD_III_IV'].map({'yes': 1, 'no': 0}).astype(int)

    # SMOTE only on survival (primary target) — X and y_survival together
    k = min(5, y_survival.value_counts().min() - 1)
    smote = SMOTE(random_state=42, k_neighbors=k)
    X_res, y_s_res = smote.fit_resample(X, y_survival)

    # For relapse and GvHD — resample using same indices via nearest neighbors
    # Simplest correct approach: fit SMOTE separately but keep X the same size
    # by resampling y_relapse and y_gvhd to match X_res length
    n_original = len(X)
    n_smoted   = len(X_res)

    # The first n_original rows of X_res are original rows (SMOTE appends synthetic)
    # So we can directly use original labels for first n_original rows
    # and repeat/sample for the synthetic rows
    np.random.seed(42)
    synthetic_indices = np.random.choice(n_original, n_smoted - n_original, replace=True)

    y_r_res = np.concatenate([
        y_relapse.values,
        y_relapse.values[synthetic_indices]
    ])
    y_g_res = np.concatenate([
        y_gvhd.values,
        y_gvhd.values[synthetic_indices]
    ])

    print(f"[4] After SMOTE: {len(X_res)} rows")
    print(f"[4] Survival dist : {pd.Series(y_s_res).value_counts().to_dict()}")
    print(f"[4] Relapse dist  : {pd.Series(y_r_res).value_counts().to_dict()}")
    print(f"[4] GvHD dist     : {pd.Series(y_g_res).value_counts().to_dict()}")

    # Gaussian noise for remaining rows needed
    rows_needed = max(0, target_rows - len(X_res))
    if rows_needed > 0:
        idx     = np.random.choice(len(X_res), rows_needed, replace=True)
        X_extra = pd.DataFrame(X_res, columns=feature_cols).iloc[idx].copy()

        noise_cols = [
            'donor_age', 'recipient_age', 'recipient_body_mass',
            'CD34_x1e6_per_kg', 'CD3_x1e8_per_kg',
            'CD3_to_CD34_ratio', 'ANC_recovery', 'PLT_recovery'
        ]
        noise_idx = [feature_cols.index(c) for c in noise_cols]
        X_extra.iloc[:, noise_idx] += np.random.normal(
            0, 0.02, (rows_needed, len(noise_idx))
        )

        X_final   = pd.concat(
            [pd.DataFrame(X_res, columns=feature_cols), X_extra],
            ignore_index=True
        )
        y_s_final = np.concatenate([y_s_res, y_s_res[idx]])
        y_r_final = np.concatenate([y_r_res, y_r_res[idx]])
        y_g_final = np.concatenate([y_g_res, y_g_res[idx]])
    else:
        X_final   = pd.DataFrame(X_res, columns=feature_cols)
        y_s_final = y_s_res
        y_r_final = y_r_res
        y_g_final = y_g_res

    print(f"[4] Final augmented size: {len(X_final)} rows")
    return X_final, y_s_final, y_r_final, y_g_final, feature_cols, encoders

# ============================================================
# STEP 5: TRAIN 3 MODELS
# ============================================================

def train_models(X, y_survival, y_relapse, y_gvhd):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

    def train_single(X, y, name):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=8,
            min_samples_leaf=5, min_samples_split=10,
            max_features='sqrt', class_weight='balanced',
            random_state=42
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc  = accuracy_score(y_test,  y_pred)
        auc       = roc_auc_score(y_test, y_prob)
        cv        = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

        print(f"\n{'='*50}")
        print(f"  {name} Model")
        print(f"{'='*50}")
        print(f"  Train Accuracy : {train_acc:.4f}")
        print(f"  Test  Accuracy : {test_acc:.4f}")
        print(f"  ROC-AUC        : {auc:.4f}")
        print(f"  5-Fold CV      : {cv.mean():.4f} +/- {cv.std():.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

        return clf

    print("\n[5] Training models...")
    m_survival = train_single(X, y_survival, "SURVIVAL STATUS")
    m_relapse  = train_single(X, y_relapse,  "RELAPSE")
    m_gvhd     = train_single(X, y_gvhd,     "GvHD (Stage III/IV)")

    return m_survival, m_relapse, m_gvhd


# ============================================================
# STEP 6: SAVE MODELS
# ============================================================

def save_models(m_survival, m_relapse, m_gvhd, feature_cols, encoders):
    import joblib, json

    joblib.dump(m_survival,  'model_survival.pkl')
    joblib.dump(m_relapse,   'model_relapse.pkl')
    joblib.dump(m_gvhd,      'model_gvhd.pkl')
    joblib.dump(encoders,    'encoders.pkl')
    with open('feature_cols.json', 'w') as f:
        json.dump(list(feature_cols), f)

    print("\n[6] Saved files:")
    print("     model_survival.pkl")
    print("     model_relapse.pkl")
    print("     model_gvhd.pkl")
    print("     encoders.pkl")
    print("     feature_cols.json")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    df = load_arff('bone-marrow.arff')  
    df = clean_data(df)
    df = engineer_features(df)
    X, y_s, y_r, y_g, feature_cols, encoders = augment_data(df, target_rows=450)
    m_survival, m_relapse, m_gvhd = train_models(X, y_s, y_r, y_g)
    save_models(m_survival, m_relapse, m_gvhd, feature_cols, encoders)
    print("\n[DONE] Run 'uvicorn api:app --reload' to start the API server.")