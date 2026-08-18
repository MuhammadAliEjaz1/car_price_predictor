"""
Clean and prepare the raw scraped PakWheels data for modeling.

Input:  data/raw/cars_raw.csv        (scraper output)
Output: data/processed/cars_clean.csv

What this does:
  1. Drops fully-empty / broken rows
  2. Parses `model` out of `title` (titles follow a consistent
     "{Make} {Model...} {Year} for sale in {City}" pattern)
  3. Converts price_pkr, mileage_km, engine_cc, year to proper numeric types
  4. Standardizes categorical text (strip whitespace, consistent casing)
  5. Reports what got dropped and why, so nothing silently disappears

Usage:
    python clean.py --input ../data/raw/cars_raw.csv --output ../data/processed/cars_clean.csv
"""

import argparse
import re

import pandas as pd


def parse_model_from_title(row):
    """
    Titles look like: "Honda Civic 2019 for sale in Karachi"
    Make is already known (scraped separately) -- strip it off the front,
    then strip the "{year} for sale in {city}" suffix. Whatever's left in
    the middle is the model (+ variant, e.g. "Civic Oriel 1.8 i-VTEC").
    """
    title = str(row.get("title", "") or "")
    make = str(row.get("make", "") or "")

    if not title or not make:
        return None

    text = title
    if text.startswith(make):
        text = text[len(make):].strip()

    # Strip everything from " {year} for sale in ..." onward
    text = re.sub(r"\s+\d{4}\s+for sale in .*$", "", text, flags=re.IGNORECASE).strip()

    return text if text else None


def to_numeric(series):
    """Strip non-digit characters (commas, 'km', 'cc', etc.) and convert to float."""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^\d.]", "", regex=True),
        errors="coerce",
    )


def clean(input_path, output_path):
    df = pd.read_csv(input_path)
    start_count = len(df)
    print(f"Loaded {start_count} rows from {input_path}")

    # --- 1. Drop rows with no title/make/price at all (broken scrapes) ---
    essential = ["title", "make", "year", "price_pkr"]
    before = len(df)
    df = df.dropna(subset=essential, how="any")
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows missing essential fields ({essential})")

    # --- 2. Deduplicate on listing_id, just in case ---
    before = len(df)
    df = df.drop_duplicates(subset=["listing_id"], keep="first")
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} duplicate listing_id rows")

    # --- 3. Parse model from title ---
    df["model"] = df.apply(parse_model_from_title, axis=1)
    missing_model = df["model"].isna().sum()
    if missing_model:
        print(f"Warning: {missing_model} rows had no parseable model (title didn't match expected pattern)")

    # --- 4. Numeric conversions ---
    df["price_pkr"] = to_numeric(df["price_pkr"])
    df["mileage_km"] = to_numeric(df["mileage_km"])
    df["engine_cc"] = to_numeric(df["engine_cc"])
    df["year"] = to_numeric(df["year"]).astype("Int64")

    # Drop rows where price or year failed to convert -- unusable for modeling
    before = len(df)
    df = df.dropna(subset=["price_pkr", "year"])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with unparseable price/year")

    # Sanity filter: drop implausible prices/years (junk listings, placeholder prices)
    before = len(df)
    df = df[(df["price_pkr"] >= 100_000) & (df["price_pkr"] <= 100_000_000)]
    df = df[(df["year"] >= 1990) & (df["year"] <= 2027)]
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with implausible price (outside 100k-100M PKR) or year (outside 1990-2027)")

    # --- 5. Standardize categorical text ---
    text_cols = ["make", "model", "fuel_type", "transmission", "city", "assembly", "registered_in", "seller_type"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": pd.NA, "": pd.NA})

    # --- 6. Drop columns not useful for modeling (kept for reference during scraping) ---
    df = df.drop(columns=["scraped_at"], errors="ignore")

    df.to_csv(output_path, index=False)
    print(f"\nWrote {len(df)} cleaned rows to {output_path}")
    print(f"Total dropped: {start_count - len(df)} ({(start_count - len(df)) / start_count * 100:.1f}%)")

    # Quick summary
    print("\n--- Missing values remaining ---")
    print(df.isna().sum()[df.isna().sum() > 0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Clean raw PakWheels scrape output")
    ap.add_argument("--input", type=str, default="../data/raw/cars_raw.csv")
    ap.add_argument("--output", type=str, default="../data/processed/cars_clean.csv")
    args = ap.parse_args()
    clean(args.input, args.output)
