"""
PakWheels Car Price Predictor - Streamlit demo.

Loads the trained XGBoost model and lets a user enter a car's details
to get a predicted fair price, plus a comparison against similar
listings actually seen in the training data.

Run with:
    streamlit run app.py
(from inside the app/ folder, or point streamlit at its full path)
"""

import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

CURRENT_YEAR = 2026

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "..", "models", "xgb_model.json")
METADATA_PATH = os.path.join(APP_DIR, "..", "models", "model_metadata.json")
FEATURES_CSV_PATH = os.path.join(APP_DIR, "..", "data", "processed", "cars_features.csv")

# Same province mapping used in scripts/features.py -- kept in sync so a
# manually entered city/registered-in value resolves the same way here as
# it did during training. If you add cities to features.py, mirror the
# change here too.
CITY_TO_PROVINCE = {
    "lahore": "Punjab", "faisalabad": "Punjab", "multan": "Punjab",
    "gujranwala": "Punjab", "rawalpindi": "Punjab", "sialkot": "Punjab",
    "sargodha": "Punjab", "bahawalpur": "Punjab", "sahiwal": "Punjab",
    "gujrat": "Punjab", "sheikhupura": "Punjab", "chakwal": "Punjab",
    "bhakkar": "Punjab", "wah cantt": "Punjab", "jhelum": "Punjab",
    "kasur": "Punjab", "okara": "Punjab", "vehari": "Punjab",
    "karachi": "Sindh", "hyderabad": "Sindh", "sukkur": "Sindh",
    "larkana": "Sindh", "nawabshah": "Sindh", "mirpurkhas": "Sindh",
    "peshawar": "KPK", "abbottabad": "KPK", "mansehra": "KPK",
    "swat": "KPK", "dir": "KPK", "khyber": "KPK", "mardan": "KPK",
    "kohat": "KPK", "swabi": "KPK",
    "quetta": "Balochistan", "gwadar": "Balochistan",
    "islamabad": "Islamabad",
    "punjab": "Punjab", "sindh": "Sindh", "kpk": "KPK", "balochistan": "Balochistan",
}


def map_to_province(value):
    return CITY_TO_PROVINCE.get(str(value).strip().lower(), "Other")


@st.cache_resource
def load_model():
    model = xgb.XGBRegressor(enable_categorical=True)
    model.load_model(MODEL_PATH)
    return model


@st.cache_data
def load_metadata():
    with open(METADATA_PATH) as f:
        return json.load(f)


@st.cache_data
def load_reference_data():
    """Used for the make->model dropdown and the 'similar listings' comparison."""
    return pd.read_csv(FEATURES_CSV_PATH)


def build_input_row(
    make, model_name, year, mileage_km, fuel_type, transmission,
    city, assembly, registered_in, engine_cc, metadata,
):
    car_age = max(CURRENT_YEAR - year, 0)
    mileage_per_year = mileage_km / max(car_age, 1)
    is_imported = 1.0 if assembly == "Imported" else 0.0
    listing_province = map_to_province(city)
    registered_province = map_to_province(registered_in)

    if registered_in == "Un-Registered":
        registered_matches_listing = "nan"
    else:
        registered_matches_listing = str(registered_province == listing_province)

    row = {
        "car_age": car_age,
        "mileage_km": mileage_km,
        "mileage_per_year": mileage_per_year,
        "engine_cc": engine_cc,
        "is_imported": is_imported,
        "make": make,
        "model": model_name,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "city": city,
        "assembly": assembly,
        "registered_in": registered_in,
        "seller_type": "Dealer",  # constant in training data -- see module note
        "listing_province": listing_province,
        "registered_province": registered_province,
        "registered_matches_listing": registered_matches_listing,
    }

    df = pd.DataFrame([row])

    # Align categorical dtypes to the exact levels seen during training,
    # so XGBoost doesn't error on a category it never saw (see scripts/train.py).
    for col in metadata["categorical_features"]:
        levels = metadata["category_levels"][col]
        df[col] = pd.Categorical(df[col], categories=levels)

    ordered_cols = metadata["numeric_features"] + metadata["categorical_features"]
    return df[ordered_cols]


def find_comparables(ref_df, make, model_name, year, tolerance_years=2):
    mask = (
        (ref_df["make"] == make)
        & (ref_df["model"] == model_name)
        & (ref_df["year"].between(year - tolerance_years, year + tolerance_years))
    )
    return ref_df.loc[mask, "price_pkr"]


def main():
    st.set_page_config(page_title="PakWheels Price Predictor", page_icon="🚗", layout="centered")
    st.title("🚗 Used Car Fair Price Predictor")
    st.caption(
        "Trained on real PakWheels.com listings, with Pakistan-specific features "
        "(import/assembly status, registered-city vs listing-city) that generic "
        "price predictors don't account for."
    )

    metadata = load_metadata()
    ref_df = load_reference_data()
    model = load_model()

    make_to_models = (
        ref_df.groupby("make")["model"]
        .apply(lambda s: sorted(s.dropna().unique()))
        .to_dict()
    )
    makes = sorted(make_to_models.keys())

    st.subheader("Car details")
    col1, col2 = st.columns(2)

    with col1:
        make = st.selectbox("Make", makes, index=makes.index("Toyota") if "Toyota" in makes else 0)
        available_models = make_to_models.get(make, [])
        model_name = st.selectbox("Model", available_models)
        year = st.number_input("Year", min_value=1990, max_value=CURRENT_YEAR, value=2018, step=1)
        mileage_km = st.number_input("Mileage (km)", min_value=0, max_value=500_000, value=60_000, step=1000)
        engine_cc = st.number_input("Engine capacity (cc)", min_value=0, max_value=6000, value=1300, step=100)

    with col2:
        fuel_type = st.selectbox("Fuel type", metadata["category_levels"]["fuel_type"])
        transmission = st.selectbox("Transmission", metadata["category_levels"]["transmission"])
        assembly = st.selectbox("Assembly", metadata["category_levels"]["assembly"])
        city = st.selectbox(
            "Listing city",
            metadata["category_levels"]["city"],
            index=metadata["category_levels"]["city"].index("Lahore")
            if "Lahore" in metadata["category_levels"]["city"] else 0,
        )
        registered_in = st.selectbox("Registered in", metadata["category_levels"]["registered_in"])

    if st.button("Predict fair price", type="primary", use_container_width=True):
        X = build_input_row(
            make, model_name, year, mileage_km, fuel_type, transmission,
            city, assembly, registered_in, engine_cc, metadata,
        )
        pred_log = model.predict(X)[0]
        predicted_price = float(np.expm1(pred_log))

        st.divider()
        st.metric("Predicted fair price", f"Rs. {predicted_price:,.0f}")

        comparables = find_comparables(ref_df, make, model_name, year)
        if len(comparables) >= 3:
            low, median, high = comparables.quantile([0.1, 0.5, 0.9])
            st.write(
                f"**Similar listings** ({len(comparables)} found, {make} {model_name}, "
                f"±2 years): priced Rs. {low:,.0f} – Rs. {high:,.0f} "
                f"(median Rs. {median:,.0f})"
            )
            if predicted_price < low:
                st.info("This predicted price is below the typical range for comparable cars.")
            elif predicted_price > high:
                st.info("This predicted price is above the typical range for comparable cars.")
            else:
                st.success("This predicted price falls within the typical range for comparable cars.")
        else:
            st.caption("Not enough similar listings in the training data for a comparison range.")

        with st.expander("Model performance (on held-out test data)"):
            m = metadata["test_metrics"]
            st.write(f"MAE: Rs. {m['mae']:,.0f}  |  RMSE: Rs. {m['rmse']:,.0f}  |  MAPE: {m['mape']:.1f}%")
            st.caption(
                "On average, predictions are off by about this much on cars the model "
                "never saw during training -- use this as a rough error margin, not an exact figure."
            )


if __name__ == "__main__":
    main()
