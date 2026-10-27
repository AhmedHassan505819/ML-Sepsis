"""
Shared preprocessing functions, imported by notebooks/scripts.
Move the split/imputation logic from scripts/step1_split_and_baseline.py
in here as functions once it's stable, so notebooks can just:

    from src.preprocessing import patient_level_split, impute_representation_a
"""
