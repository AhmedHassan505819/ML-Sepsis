"""
Mid-semester pipeline, Step 1
==============================
1. Patient-level train/val/test split, stratified by whether the patient
   ever develops sepsis (prevents leakage AND keeps class balance similar
   across splits).
2. Representation A: leakage-safe imputation (forward-fill per patient,
   then fill remaining gaps using statistics computed ONLY on the train
   set - never fit imputation on val/test).
3. Two baseline models (Logistic Regression, HistGradientBoostingClassifier)
   trained on Representation A, evaluated with PR-AUC and F1.

Run this in Colab after mounting Drive and loading sepsis_merged.csv.
Only needs pandas, numpy, scikit-learn (all preinstalled in Colab).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 42
DATA_PATH = "sepsis_merged.csv"   # update path if loading from Drive, e.g. "/content/drive/MyDrive/.../sepsis_merged.csv"

# ----------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------

print("[load] reading merged dataset...")
df = pd.read_csv(DATA_PATH)
print(f"[load] {len(df):,} rows, {df['patient_id'].nunique():,} patients")

# ----------------------------------------------------------------------
# 2. PATIENT-LEVEL STRATIFIED SPLIT (70 / 15 / 15)
# ----------------------------------------------------------------------
# Why patient-level: rows from the same patient are highly correlated.
# A random row-level split lets the model "see" a patient's future in
# training and predict their past in test -> inflated, fake results.
# Why stratified: sepsis patients are only ~7% of all patients, so a
# naive random split of patient IDs could easily over/under-represent
# them in one split by chance.

print("\n[split] building patient-level stratified split...")
patient_labels = df.groupby("patient_id")["SepsisLabel"].max().reset_index()
patient_labels.columns = ["patient_id", "ever_septic"]

train_ids, temp_ids = train_test_split(
    patient_labels, test_size=0.30,
    stratify=patient_labels["ever_septic"], random_state=RANDOM_STATE
)
val_ids, test_ids = train_test_split(
    temp_ids, test_size=0.50,
    stratify=temp_ids["ever_septic"], random_state=RANDOM_STATE
)

train_df = df[df.patient_id.isin(train_ids.patient_id)].copy()
val_df = df[df.patient_id.isin(val_ids.patient_id)].copy()
test_df = df[df.patient_id.isin(test_ids.patient_id)].copy()

for name, split_df, id_df in [("train", train_df, train_ids),
                               ("val", val_df, val_ids),
                               ("test", test_df, test_ids)]:
    pct_septic_patients = 100 * id_df["ever_septic"].mean()
    print(f"  {name:5s}: {split_df.patient_id.nunique():5d} patients, "
          f"{len(split_df):8,d} rows, {pct_septic_patients:.2f}% ever-septic patients")

# sanity check: no patient_id should appear in more than one split
assert set(train_df.patient_id) & set(val_df.patient_id) == set()
assert set(train_df.patient_id) & set(test_df.patient_id) == set()
assert set(val_df.patient_id) & set(test_df.patient_id) == set()
print("  [ok] no patient overlap between splits")

# ----------------------------------------------------------------------
# 3. REPRESENTATION A: LEAKAGE-SAFE IMPUTATION
# ----------------------------------------------------------------------
# Step A: forward-fill within each patient (carry last known value forward)
# Step B: any values still missing (e.g. nothing measured yet for that
#         patient) get filled with the column MEDIAN COMPUTED ON TRAIN ONLY.
#         Fitting this on val/test would leak population statistics across
#         the split boundary.

NON_FEATURE_COLS = ["patient_id", "SepsisLabel"]  # keep hospital/hour/ICULOS as features

def forward_fill_per_patient(data):
    data = data.sort_values(["patient_id", "hour"])
    data = data.groupby("patient_id", group_keys=False).apply(
        lambda g: g.ffill()
    )
    return data

print("\n[impute] forward-filling within each patient...")
train_df = forward_fill_per_patient(train_df)
val_df = forward_fill_per_patient(val_df)
test_df = forward_fill_per_patient(test_df)

feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
numeric_cols = [c for c in feature_cols if c != "hospital"]

print("[impute] filling remaining NaNs using TRAIN-set medians only...")
train_medians = train_df[numeric_cols].median()
train_df[numeric_cols] = train_df[numeric_cols].fillna(train_medians)
val_df[numeric_cols] = val_df[numeric_cols].fillna(train_medians)
test_df[numeric_cols] = test_df[numeric_cols].fillna(train_medians)

# one-hot encode hospital (A/B) - fit categories on train, apply same to val/test
train_df = pd.get_dummies(train_df, columns=["hospital"], prefix="hosp")
val_df = pd.get_dummies(val_df, columns=["hospital"], prefix="hosp")
test_df = pd.get_dummies(test_df, columns=["hospital"], prefix="hosp")
for col in [c for c in train_df.columns if c.startswith("hosp_")]:
    if col not in val_df.columns:
        val_df[col] = 0
    if col not in test_df.columns:
        test_df[col] = 0

# save processed splits so you don't have to redo this every session
train_df.to_csv("repA_train.csv", index=False)
val_df.to_csv("repA_val.csv", index=False)
test_df.to_csv("repA_test.csv", index=False)
print("[saved] repA_train.csv, repA_val.csv, repA_test.csv")

# ----------------------------------------------------------------------
# 4. BASELINE MODELS ON REPRESENTATION A
# ----------------------------------------------------------------------

model_feature_cols = [c for c in train_df.columns
                       if c not in ("patient_id", "SepsisLabel")]

X_train, y_train = train_df[model_feature_cols], train_df["SepsisLabel"]
X_val, y_val = val_df[model_feature_cols], val_df["SepsisLabel"]

sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

print("\n" + "=" * 60)
print("BASELINE 1: Logistic Regression")
print("=" * 60)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
logreg.fit(X_train_scaled, y_train, sample_weight=sample_weight)

val_probs_lr = logreg.predict_proba(X_val_scaled)[:, 1]
val_preds_lr = (val_probs_lr >= 0.5).astype(int)

pr_auc_lr = average_precision_score(y_val, val_probs_lr)
f1_lr = f1_score(y_val, val_preds_lr)
print(f"PR-AUC: {pr_auc_lr:.4f}")
print(f"F1 (threshold=0.5): {f1_lr:.4f}")
print(classification_report(y_val, val_preds_lr, digits=3))

print("\n" + "=" * 60)
print("BASELINE 2: HistGradientBoostingClassifier")
print("=" * 60)
hgb = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
hgb.fit(X_train, y_train, sample_weight=sample_weight)

val_probs_hgb = hgb.predict_proba(X_val)[:, 1]
val_preds_hgb = (val_probs_hgb >= 0.5).astype(int)

pr_auc_hgb = average_precision_score(y_val, val_probs_hgb)
f1_hgb = f1_score(y_val, val_preds_hgb)
print(f"PR-AUC: {pr_auc_hgb:.4f}")
print(f"F1 (threshold=0.5): {f1_hgb:.4f}")
print(classification_report(y_val, val_preds_hgb, digits=3))

print("\n" + "=" * 60)
print("SUMMARY - REPRESENTATION A (copy into your report's results table)")
print("=" * 60)
print(f"{'Model':<30}{'PR-AUC':>10}{'F1':>10}")
print(f"{'Logistic Regression':<30}{pr_auc_lr:>10.4f}{f1_lr:>10.4f}")
print(f"{'HistGradientBoosting':<30}{pr_auc_hgb:>10.4f}{f1_hgb:>10.4f}")

print("""
NEXT STEPS (not in this script):
1. Download PhysioNet's official evaluate_sepsis_score.py and run it on
   these predictions to get the Challenge utility score - don't
   reimplement the scoring logic yourself, use the official one.
2. Build Representation B: add a binary "<col>_measured" mask column for
   each lab/vital BEFORE forward-filling (1 if the raw value was present
   at that hour, 0 if it was NaN and about to be filled). Retrain both
   models on train+mask features, compare PR-AUC/F1 against this table.
3. Keep test_df untouched until the final paper - only use val_df for
   all comparisons until then.
""")
