"""
PhysioNet / CinC Challenge 2019 - Early Sepsis Prediction
Data download + loading + basic EDA for synopsis writing.

Run this in Google Colab or locally (needs internet access).
Requires: pandas, numpy, requests  (pip install pandas numpy requests)

WHAT THIS SCRIPT DOES
1. Downloads the training data (Hospital A + Hospital B) from PhysioNet.
2. Loads all per-patient .psv files into a single DataFrame.
3. Prints the stats you need for the synopsis:
   - number of patients (total, per hospital)
   - % of patients who ever develop sepsis
   - % of all hourly rows labeled positive (class imbalance)
   - missingness per column
   - basic row/column counts
4. Saves one merged CSV so you don't have to redo this.

DOWNLOAD METHOD (confirmed against physionet.org/content/challenge-2019/1.0.0/):
There is no single zip file anymore. This script pulls the data straight from
PhysioNet's public AWS S3 bucket via `aws s3 sync` (fast, no AWS account
needed), and falls back to `wget -r` mirroring if that's unavailable.
If both fail in your environment, run one of these by hand in a terminal
or Colab cell (prefix with `!` in Colab), then just rerun this script -
it will detect the existing .psv files and skip straight to loading:
    !aws s3 sync --no-sign-request s3://physionet-open/challenge-2019/1.0.0/training/ data
    !wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/training/
"""

import os
import subprocess
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# 1. DOWNLOAD
# ----------------------------------------------------------------------
# Confirmed 2026-09 from https://physionet.org/content/challenge-2019/1.0.0/ :
# PhysioNet no longer serves a single training_setX.zip. The two supported
# ways to fetch the data are:
#   (a) AWS S3 sync (fast, resumable, no AWS account needed - public bucket)
#       aws s3 sync --no-sign-request s3://physionet-open/challenge-2019/1.0.0/training/ DEST
#   (b) wget mirroring the file tree
#       wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/training/
# This script tries (a) first (much faster for ~40,000 small files), and
# falls back to (b) if awscli isn't available or the sync fails.

DATA_DIR = "data"
S3_PREFIX = "s3://physionet-open/challenge-2019/1.0.0/training/"
HTTP_PREFIX = "https://physionet.org/files/challenge-2019/1.0.0/training/"

def download_via_s3():
    print("[download] trying AWS S3 sync (no credentials needed, public bucket)...")
    subprocess.run(["pip", "install", "-q", "awscli"], check=False)
    os.makedirs(DATA_DIR, exist_ok=True)
    result = subprocess.run(
        ["aws", "s3", "sync", "--no-sign-request", S3_PREFIX, DATA_DIR],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("[!] S3 sync failed:\n", result.stderr[-1000:])
        return False
    print("[done] S3 sync complete")
    return True

def download_via_wget():
    print("[download] falling back to wget mirroring...")
    os.makedirs(DATA_DIR, exist_ok=True)
    result = subprocess.run(
        ["wget", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=5",
         "-P", DATA_DIR, HTTP_PREFIX],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("[!] wget failed too:\n", result.stderr[-1000:])
        return False
    print("[done] wget mirror complete")
    return True

def try_download_all():
    if os.path.isdir(DATA_DIR) and any(
        f.endswith(".psv") for _, _, files in os.walk(DATA_DIR) for f in files
    ):
        print(f"[skip] .psv files already found under {DATA_DIR}/")
        return
    if not download_via_s3():
        if not download_via_wget():
            print("\n[!] Both automatic methods failed.")
            print("    Run this manually in a terminal/Colab cell:")
            print(f"    !aws s3 sync --no-sign-request {S3_PREFIX} {DATA_DIR}")
            print("    or")
            print(f"    !wget -r -N -c -np {HTTP_PREFIX}")

# ----------------------------------------------------------------------
# 2. LOAD ALL .psv FILES INTO ONE DATAFRAME
# ----------------------------------------------------------------------

def load_hospital(folder_path, hospital_label):
    """Reads every patient .psv file in a folder into one long DataFrame."""
    rows = []
    files = [f for f in os.listdir(folder_path) if f.endswith(".psv")]
    print(f"[load] {hospital_label}: {len(files)} patient files found")

    for fname in files:
        patient_id = fname.replace(".psv", "")
        df = pd.read_csv(os.path.join(folder_path, fname), sep="|")
        df["patient_id"] = patient_id
        df["hospital"] = hospital_label
        df["hour"] = np.arange(len(df))  # hour index within stay
        rows.append(df)

    return pd.concat(rows, ignore_index=True)

def load_all_data():
    frames = []
    for name, hosp_label in [("training_setA", "A"), ("training_setB", "B")]:
        folder = os.path.join(DATA_DIR, name)
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Expected folder not found: {folder}\n"
                f"Check that the download step completed - look for .psv files "
                f"somewhere under ./{DATA_DIR}/ and adjust the path above if needed."
            )
        frames.append(load_hospital(folder, hosp_label))
    full = pd.concat(frames, ignore_index=True)
    return full

# ----------------------------------------------------------------------
# 3. EDA STATS FOR THE SYNOPSIS
# ----------------------------------------------------------------------

def run_eda(df):
    print("\n" + "=" * 60)
    print("BASIC SHAPE")
    print("=" * 60)
    print(f"Total rows (patient-hours): {len(df):,}")
    print(f"Total columns: {df.shape[1]}")
    print(f"Columns: {list(df.columns)}")

    print("\n" + "=" * 60)
    print("PATIENT COUNTS")
    print("=" * 60)
    n_patients = df["patient_id"].nunique()
    per_hosp = df.groupby("hospital")["patient_id"].nunique()
    print(f"Total unique patients: {n_patients:,}")
    print(per_hosp)

    print("\n" + "=" * 60)
    print("CLASS IMBALANCE")
    print("=" * 60)
    positive_rows = df["SepsisLabel"].sum()
    print(f"Positive (SepsisLabel=1) rows: {positive_rows:,} "
          f"({100 * positive_rows / len(df):.2f}% of all rows)")

    patients_ever_septic = df.groupby("patient_id")["SepsisLabel"].max()
    pct_patients_septic = 100 * patients_ever_septic.mean()
    print(f"Patients who ever develop sepsis: {patients_ever_septic.sum():,} "
          f"({pct_patients_septic:.2f}% of all patients)")

    print("\n" + "=" * 60)
    print("MISSINGNESS PER COLUMN (top 15 worst)")
    print("=" * 60)
    missing_pct = df.isna().mean().sort_values(ascending=False) * 100
    print(missing_pct.head(15).round(2))

    print("\n" + "=" * 60)
    print("STAY LENGTH (hours per patient)")
    print("=" * 60)
    stay_len = df.groupby("patient_id")["hour"].max() + 1
    print(stay_len.describe())

    return {
        "n_rows": len(df),
        "n_patients": n_patients,
        "pct_positive_rows": 100 * positive_rows / len(df),
        "pct_patients_septic": pct_patients_septic,
        "missing_pct": missing_pct,
    }

# ----------------------------------------------------------------------
# 4. MAIN
# ----------------------------------------------------------------------

if __name__ == "__main__":
    try_download_all()
    full_df = load_all_data()

    out_csv = "sepsis_merged.csv"
    full_df.to_csv(out_csv, index=False)
    print(f"\n[saved] merged dataset written to {out_csv} "
          f"({os.path.getsize(out_csv) / 1e6:.1f} MB)")

    stats = run_eda(full_df)

    print("\n" + "=" * 60)
    print("COPY THESE NUMBERS INTO YOUR SYNOPSIS")
    print("=" * 60)
    print(f"- Total patients: {stats['n_patients']:,}")
    print(f"- Total patient-hours: {stats['n_rows']:,}")
    print(f"- % patients who develop sepsis: {stats['pct_patients_septic']:.2f}%")
    print(f"- % of hourly rows labeled positive: {stats['pct_positive_rows']:.2f}%")
    print(f"- Worst missingness columns:\n{stats['missing_pct'].head(5)}")
