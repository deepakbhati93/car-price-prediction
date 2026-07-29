import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "cardetails.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, "feature_columns.json")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.json")

NUMERIC_FEATURES = ["km_driven", "car_age"]
REFERENCE_CATEGORIES = {
    "fuel": "CNG",
    "seller_type": "Dealer",
    "transmission": "Automatic",
    "owner": "First Owner",
}


def train_and_save_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().reset_index(drop=True)

    df["car_name"] = df["name"].astype(str)

    current_year = datetime.now().year
    df["car_age"] = current_year - df["year"].astype(int)
    df["selling_price_log"] = np.log1p(df["selling_price"].astype(float))

    categorical_cols = ["car_name", "fuel", "seller_type", "transmission", "owner"]
    feature_columns = ["km_driven", "car_age", "car_name", "fuel", "seller_type", "transmission", "owner"]
    df_features = df[feature_columns].copy()
    df_features = pd.get_dummies(df_features, columns=categorical_cols, drop_first=False)
    df_features["selling_price_log"] = df["selling_price_log"]

    scaler = StandardScaler()
    df_features[NUMERIC_FEATURES] = scaler.fit_transform(df_features[NUMERIC_FEATURES])

    X = df_features.drop(columns=["selling_price_log"])
    y = df_features["selling_price_log"]

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with open(FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    metadata = {
        "car_names": sorted(df["car_name"].astype(str).dropna().unique().tolist()),
        "fuel_types": sorted(df["fuel"].astype(str).unique().tolist()),
        "seller_types": sorted(df["seller_type"].astype(str).unique().tolist()),
        "transmissions": sorted(df["transmission"].astype(str).unique().tolist()),
        "owner_types": sorted(df["owner"].astype(str).unique().tolist()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "km_driven_max": int(df["km_driven"].max()),
        "reference_categories": REFERENCE_CATEGORIES,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Model trained and saved successfully.")
    print(f"Model file: {MODEL_PATH}")
    print(f"Scaler file: {SCALER_PATH}")
    print(f"Feature columns file: {FEATURE_COLUMNS_PATH}")
    print(f"Metadata file: {METADATA_PATH}")


if __name__ == "__main__":
    train_and_save_model()
