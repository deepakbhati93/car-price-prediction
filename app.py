import json
import os
import subprocess
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit as st

st.set_page_config(
    page_title="Car Price Prediction", # Changes the tab title text
    page_icon=" 🚗 ",                         # Optional: changes the icon next to the text
    layout="wide"                          # Optional: sets page width
)

# Rest of your app code goes below...
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, "feature_columns.json")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.json")

NUMERIC_FEATURES = ["km_driven", "car_age"]
CATEGORICAL_COLUMNS = ["fuel", "seller_type", "transmission", "owner"]

st.set_page_config(page_title="Car Price Prediction", layout="centered")
st.title("🚗 Car Price Prediction")
st.write("Enter the car details below to get a predicted resale price.")


@st.cache_resource
def load_artifacts():
    if not all(
        [
            os.path.exists(MODEL_PATH),
            os.path.exists(SCALER_PATH),
            os.path.exists(FEATURE_COLUMNS_PATH),
            os.path.exists(METADATA_PATH),
        ]
    ):
        if os.environ.get("SKIP_TRAIN") == "1":
            st.error(
                "Required model artifacts not found. Set SKIP_TRAIN!=1 locally or upload pre-trained artifacts (model.pkl, scaler.pkl, feature_columns.json, metadata.json) when deploying to Streamlit Cloud."
            )
            st.stop()
        with st.spinner("Training the model and preparing the app..."):
            try:
                subprocess.run(
                    [sys.executable, os.path.join(BASE_DIR, "train_model.py")],
                    cwd=BASE_DIR,
                    check=True,
                    text=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as exc:
                error_output = exc.stderr.strip() or exc.stdout.strip() or str(exc)
                st.error(f"Model training failed: {error_output}")
                st.stop()

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, scaler, feature_columns, metadata


model, scaler, feature_columns, metadata = load_artifacts()


def load_dataset_options():
    dataset_path = os.path.join(BASE_DIR, "cardetails.csv")
    if not os.path.exists(dataset_path):
        return metadata

    df = pd.read_csv(dataset_path)
    df = df.drop_duplicates().reset_index(drop=True)

    options = {
        "car_names": sorted(df["name"].astype(str).dropna().unique().tolist()),
        "fuel_types": sorted(df["fuel"].astype(str).unique().tolist()),
        "seller_types": sorted(df["seller_type"].astype(str).unique().tolist()),
        "transmissions": sorted(df["transmission"].astype(str).unique().tolist()),
        "owner_types": sorted(df["owner"].astype(str).unique().tolist()),
    }
    return options


def get_model_compatible_options(column_name, dataset_values):
    prefix = f"{column_name}_"
    feature_values = [
        col[len(prefix):]
        for col in feature_columns
        if col.startswith(prefix)
    ]
    reference_value = metadata.get("reference_categories", {}).get(column_name)
    compatible_values = []
    for value in dataset_values:
        if value in feature_values or value == reference_value:
            compatible_values.append(value)
    return sorted(set(compatible_values))


def build_feature_row(car_name, fuel, seller_type, transmission, owner, year, km_driven):
    current_year = datetime.now().year
    car_age = current_year - int(year)
    if int(year) > current_year:
        raise ValueError("Manufacturing year cannot be in the future.")
    if float(km_driven) < 0:
        raise ValueError("Kilometers driven cannot be negative.")

    X_new = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    X_new.loc[0, "km_driven"] = float(km_driven)
    X_new.loc[0, "car_age"] = car_age

    categorical_values = {
        "car_name": car_name,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
    }

    for column_name, category_value in categorical_values.items():
        feature_name = f"{column_name}_{category_value}"
        if feature_name in feature_columns:
            X_new.loc[0, feature_name] = 1.0
        else:
            reference_value = metadata.get("reference_categories", {}).get(column_name)
            if reference_value is None or category_value != reference_value:
                raise ValueError(
                    f"Unsupported category '{category_value}' for '{column_name}'."
                )

    X_new[NUMERIC_FEATURES] = scaler.transform(X_new[NUMERIC_FEATURES])
    return X_new


dataset_options = load_dataset_options()
car_names = [
    name
    for name in dataset_options.get("car_names", metadata.get("car_names", []))
    if f"car_name_{name}" in feature_columns
]
if not car_names:
    car_names = metadata.get("car_names", [])
fuel_options = get_model_compatible_options("fuel", dataset_options.get("fuel_types", metadata.get("fuel_types", [])))
seller_options = get_model_compatible_options("seller_type", dataset_options.get("seller_types", metadata.get("seller_types", [])))
transmission_options = get_model_compatible_options("transmission", dataset_options.get("transmissions", metadata.get("transmissions", [])))
owner_options = get_model_compatible_options("owner", dataset_options.get("owner_types", metadata.get("owner_types", [])))

# Load dataset once and build a mapping from car name -> default attributes
dataset_path = os.path.join(BASE_DIR, "cardetails.csv")
dataset_df = None
car_defaults = {}
if os.path.exists(dataset_path):
    try:
        _df = pd.read_csv(dataset_path)
        _df = _df.drop_duplicates().reset_index(drop=True)
        dataset_df = _df
        for _, row in _df.iterrows():
            name = str(row.get("name", ""))
            car_defaults[name] = {
                "fuel": row.get("fuel", ""),
                "seller_type": row.get("seller_type", ""),
                "transmission": row.get("transmission", ""),
                "owner": row.get("owner", ""),
                "year": int(row.get("year", metadata.get("year_max", 2020))) if not pd.isna(row.get("year")) else metadata.get("year_max", 2020),
            }
    except Exception:
        dataset_df = None

if "car_name_value" not in st.session_state:
    st.session_state.car_name_value = car_names[0] if car_names else ""
if "fuel_value" not in st.session_state:
    st.session_state.fuel_value = fuel_options[0] if fuel_options else "Petrol"
if "seller_value" not in st.session_state:
    st.session_state.seller_value = seller_options[0] if seller_options else "Individual"
if "transmission_value" not in st.session_state:
    st.session_state.transmission_value = transmission_options[0] if transmission_options else "Manual"
if "owner_value" not in st.session_state:
    st.session_state.owner_value = owner_options[0] if owner_options else "First Owner"
if "year_value" not in st.session_state:
    st.session_state.year_value = metadata.get("year_max", 2020) - 5

col1, col2 = st.columns(2)

with col1:
    car_name = st.selectbox("Car Name", car_names, key="car_name_value")

    if car_name in car_defaults:
        defaults = car_defaults[car_name]
        default_fuel = str(defaults.get("fuel", fuel_options[0] if fuel_options else "Petrol"))
        default_seller = str(defaults.get("seller_type", seller_options[0] if seller_options else "Individual"))
        default_transmission = str(defaults.get("transmission", transmission_options[0] if transmission_options else "Manual"))
        default_owner = str(defaults.get("owner", owner_options[0] if owner_options else "First Owner"))
        default_year = int(defaults.get("year", metadata.get("year_max", 2020)))
    else:
        default_fuel = fuel_options[0] if fuel_options else "Petrol"
        default_seller = seller_options[0] if seller_options else "Individual"
        default_transmission = transmission_options[0] if transmission_options else "Manual"
        default_owner = owner_options[0] if owner_options else "First Owner"
        default_year = metadata.get("year_max", 2020) - 5

    if st.session_state.car_name_value != car_name:
        st.session_state.car_name_value = car_name
        st.session_state.fuel_value = default_fuel if default_fuel in fuel_options else fuel_options[0] if fuel_options else "Petrol"
        st.session_state.seller_value = default_seller if default_seller in seller_options else seller_options[0] if seller_options else "Individual"
        st.session_state.transmission_value = default_transmission if default_transmission in transmission_options else transmission_options[0] if transmission_options else "Manual"
        st.session_state.owner_value = default_owner if default_owner in owner_options else owner_options[0] if owner_options else "First Owner"
        st.session_state.year_value = default_year

    fuel = st.selectbox(
        "Fuel Type",
        fuel_options,
        key="fuel_value",
        index=fuel_options.index(st.session_state.fuel_value) if st.session_state.fuel_value in fuel_options else 0,
    )
    seller_type = st.selectbox(
        "Seller Type",
        seller_options,
        key="seller_value",
        index=seller_options.index(st.session_state.seller_value) if st.session_state.seller_value in seller_options else 0,
    )
    transmission = st.selectbox(
        "Transmission",
        transmission_options,
        key="transmission_value",
        index=transmission_options.index(st.session_state.transmission_value) if st.session_state.transmission_value in transmission_options else 0,
    )

with col2:
    owner = st.selectbox(
        "Ownership",
        owner_options,
        key="owner_value",
        index=owner_options.index(st.session_state.owner_value) if st.session_state.owner_value in owner_options else 0,
    )
    year = st.number_input(
        "Manufacturing Year",
        min_value=metadata.get("year_min", 1990),
        max_value=metadata.get("year_max", 2020),
        value=st.session_state.get("year_value", metadata.get("year_max", 2020) - 5),
        step=1,
        key="year_value",
    )
    km_driven = st.number_input(
        "Kilometers Driven",
        min_value=0,
        max_value=metadata.get("km_driven_max", 1000000),
        value=45000,
        step=1000,
        key="km_driven_value",
    )

current_inputs = {
    "car_name": car_name,
    "fuel": fuel,
    "seller_type": seller_type,
    "transmission": transmission,
    "owner": owner,
    "year": year,
    "km_driven": km_driven,
}

try:
    X_new = build_feature_row(car_name, fuel, seller_type, transmission, owner, year, km_driven)
    log_price_pred = model.predict(X_new)[0]
    price_pred = float(np.expm1(log_price_pred))
    st.success(f"### Predicted Selling Price: ₹{max(price_pred, 0.0):,.0f}")
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
