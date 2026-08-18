"""
Feature engineering for the PakWheels price predictor.

Input:  data/processed/cars_clean.csv
Output: data/processed/cars_features.csv

Builds the Pakistan-specific features that differentiate this project
from a generic car-price predictor:

  - car_age               : listing year -> age in years
  - mileage_per_year       : mileage_km / car_age (wear rate, not just raw km)
  - price_per_cc            : price_pkr / engine_cc (normalizes for engine size)
  - is_imported             : binary flag from `assembly`
  - listing_province        : `city` mapped to its province
  - registered_province     : `registered_in` mapped to a province (it's
                               already a mix of cities AND provinces in the
                               raw data, so both get normalized to the same
                               scale before comparing)
  - registered_matches_listing : whether the car is registered in the same
                               province it's being sold in (True/False/NA
                               for Un-Registered cars, where the comparison
                               doesn't apply)

Usage:
    python features.py --input ../data/processed/cars_clean.csv --output ../data/processed/cars_features.csv
"""

import argparse

import pandas as pd

CURRENT_YEAR = 2026

# Maps known Pakistani cities to their province/territory. `registered_in`
# in the raw data is a mix of provinces (e.g. "Punjab") and cities (e.g.
# "Lahore") -- this dict normalizes both `city` and `registered_in` onto
# the same province-level scale so they can be compared directly.
CITY_TO_PROVINCE = {
    # Punjab
    "lahore": "Punjab", "faisalabad": "Punjab", "multan": "Punjab",
    "gujranwala": "Punjab", "rawalpindi": "Punjab", "sialkot": "Punjab",
    "sargodha": "Punjab", "bahawalpur": "Punjab", "sahiwal": "Punjab",
    "gujrat": "Punjab", "sheikhupura": "Punjab", "chakwal": "Punjab",
    "bhakkar": "Punjab", "wah cantt": "Punjab", "jhelum": "Punjab",
    "kasur": "Punjab", "okara": "Punjab", "vehari": "Punjab",
    # Sindh
    "karachi": "Sindh", "hyderabad": "Sindh", "sukkur": "Sindh",
    "larkana": "Sindh", "nawabshah": "Sindh", "mirpurkhas": "Sindh",
    # KPK
    "peshawar": "KPK", "abbottabad": "KPK", "mansehra": "KPK",
    "swat": "KPK", "dir": "KPK", "khyber": "KPK", "mardan": "KPK",
    "kohat": "KPK", "swabi": "KPK",
    # Balochistan
    "quetta": "Balochistan", "gwadar": "Balochistan",
    # Federal
    "islamabad": "Islamabad",
    # Already province-level values that appear as-is in registered_in
    "punjab": "Punjab", "sindh": "Sindh", "kpk": "KPK",
    "balochistan": "Balochistan",
}


def map_to_province(value):
    if pd.isna(value):
        return pd.NA
    key = str(value).strip().lower()
    return CITY_TO_PROVINCE.get(key, "Other")


def build_features(input_path, output_path):
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    # --- Car age & mileage rate ---
    df["car_age"] = (CURRENT_YEAR - df["year"]).clip(lower=0)
    # Avoid divide-by-zero for brand-new cars (age 0) -- treat as age 1 for the rate calc
    df["mileage_per_year"] = df["mileage_km"] / df["car_age"].replace(0, 1)

    # --- Price per cc (normalizes for engine size) ---
    # Electric vehicles have engine_cc == 0 (no displacement) -- price_per_cc
    # is meaningless for them, so leave it as NA rather than a division error.
    df["price_per_cc"] = df["price_pkr"] / df["engine_cc"].replace(0, pd.NA)

    # --- Assembly: binary flag ---
    df["is_imported"] = df["assembly"].map({"Imported": 1, "Local": 0})

    # --- Province-level location features ---
    df["listing_province"] = df["city"].apply(map_to_province)
    df["registered_province"] = df["registered_in"].apply(map_to_province)

    def compare_provinces(row):
        unregistered = row["registered_in"] == "Un-Registered"
        no_province = pd.isna(row["registered_province"]) or pd.isna(row["listing_province"])
        if unregistered or no_province:
            return pd.NA
        return row["registered_province"] == row["listing_province"]

    df["registered_matches_listing"] = df.apply(compare_provinces, axis=1)

    # --- Report unmapped locations so they can be added to the dict later ---
    unmapped_cities = set(df.loc[df["listing_province"] == "Other", "city"].dropna().unique())
    unmapped_registered = set(df.loc[df["registered_province"] == "Other", "registered_in"].dropna().unique())
    if unmapped_cities:
        print(f"\nWarning: {len(unmapped_cities)} city values not in CITY_TO_PROVINCE map:")
        print(f"  {sorted(unmapped_cities)}")
    if unmapped_registered:
        print(f"Warning: {len(unmapped_registered)} registered_in values not in CITY_TO_PROVINCE map:")
        print(f"  {sorted(unmapped_registered)}")

    df.to_csv(output_path, index=False)
    print(f"\nWrote {len(df)} rows with engineered features to {output_path}")

    # --- Quick sanity summary ---
    print("\n--- New feature summaries ---")
    print(f"is_imported: {df['is_imported'].value_counts(dropna=False).to_dict()}")
    print(f"registered_matches_listing: {df['registered_matches_listing'].value_counts(dropna=False).to_dict()}")
    print(f"car_age: min={df['car_age'].min()}, max={df['car_age'].max()}, mean={df['car_age'].mean():.1f}")
    print(f"price_per_cc: min={df['price_per_cc'].min():.0f}, max={df['price_per_cc'].max():.0f}, "
          f"mean={df['price_per_cc'].mean():.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Engineer Pakistan-specific features for the price predictor")
    ap.add_argument("--input", type=str, default="../data/processed/cars_clean.csv")
    ap.add_argument("--output", type=str, default="../data/processed/cars_features.csv")
    args = ap.parse_args()
    build_features(args.input, args.output)
