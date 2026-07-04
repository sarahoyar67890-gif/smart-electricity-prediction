"""
Data Preprocessing Module
Handles: missing values, duplicates, outlier detection, encoding, scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "electricity_dataset.csv")
CLEAN_PATH = os.path.join(PROJECT_ROOT, "data", "electricity_dataset_clean.csv")
ENCODERS_PATH = os.path.join(PROJECT_ROOT, "models", "label_encoders.pkl")


def load_data(path=RAW_PATH):
    return pd.read_csv(path)


def handle_missing_values(df):
    """Impute numeric missing values with column median."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    return df


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate rows")
    return df


def handle_outliers(df, target_col="Daily_Electricity_Consumption_kWh"):
    """Cap outliers in target using IQR method (winsorizing, not deleting)."""
    q1 = df[target_col].quantile(0.25)
    q3 = df[target_col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_outliers = ((df[target_col] < lower) | (df[target_col] > upper)).sum()
    print(f"Detected {n_outliers} outliers in target (capped via IQR)")
    df[target_col] = df[target_col].clip(lower, upper)
    return df


def encode_categoricals(df):
    """Label-encode categorical columns and persist encoders for later use."""
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le
    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    joblib.dump(encoders, ENCODERS_PATH)
    return df, encoders


def run_preprocessing():
    df = load_data()
    print(f"Loaded raw data: {df.shape}")

    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    df, encoders = encode_categoricals(df)

    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved cleaned dataset: {df.shape} -> {CLEAN_PATH}")
    return df


if __name__ == "__main__":
    run_preprocessing()
