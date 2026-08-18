"""
Train and evaluate the car price prediction model.

Input:  data/processed/cars_features.csv
Output: models/xgb_model.json, models/model_metadata.json

Trains two models for comparison:
  1. Linear Regression baseline (sanity check -- if XGBoost can't beat
     this by a solid margin, something's wrong with the setup)
  2. XGBoost (the real model) using native categorical support

IMPORTANT: `price_per_cc` is excluded from training features. It's
computed as price_pkr / engine_cc -- since it's derived directly from
the target variable, using it as a predictor would be target leakage
(the model would partially "see" the answer). It's fine to use for
post-hoc analysis/reporting, just not as a model input.

Target transform: price is right-skewed (a few very expensive cars
pull the distribution), so we train on log1p(price) and convert back
with expm1 for evaluation -- this keeps errors from being dominated by
the handful of very expensive listings.

Usage:
    python train.py --input ../data/processed/cars_features.csv
"""

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Features used for training. price_per_cc is deliberately excluded (target leakage -- see module docstring).
NUMERIC_FEATURES = ["car_age", "mileage_km", "mileage_per_year", "engine_cc", "is_imported"]
CATEGORICAL_FEATURES = [
    "make", "model", "fuel_type", "transmission", "city",
    "assembly", "registered_in", "seller_type",
    "listing_province", "registered_province", "registered_matches_listing",
]
TARGET = "price_pkr"


def load_data(path):
    df = pd.read_csv(path)

    # Drop rows with no target -- shouldn't exist post-cleaning, but be safe
    df = df.dropna(subset=[TARGET])

    # registered_matches_listing is a nullable bool -- cast to string category
    # ("True"/"False"/"nan") so both sklearn's OneHotEncoder and XGBoost's
    # categorical handling treat missing as its own valid category rather
    # than erroring.
    df["registered_matches_listing"] = df["registered_matches_listing"].astype(str)

    return df


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    print(f"\n--- {label} ---")
    print(f"MAE:  Rs. {mae:,.0f}")
    print(f"RMSE: Rs. {rmse:,.0f}")
    print(f"MAPE: {mape:.1f}%")
    return {"mae": mae, "rmse": rmse, "mape": mape}


def train_baseline(X_train, X_test, y_train_log, y_test):
    """Linear Regression on one-hot encoded features -- sanity-check baseline."""
    preprocessor = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CATEGORICAL_FEATURES),
    ])
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", LinearRegression()),
    ])
    pipeline.fit(X_train, y_train_log)
    preds_log = pipeline.predict(X_test)
    preds = np.expm1(preds_log)
    return evaluate(y_test, preds, "Baseline: Linear Regression")


def train_xgboost(X_train, X_test, y_train_log, y_test):
    """XGBoost with native categorical feature support -- the real model."""
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train_log)
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    metrics = evaluate(y_test, preds, "XGBoost")

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\n--- Top 10 feature importances ---")
    print(importances.head(10).to_string())

    return model, metrics, importances


def main(input_path, model_dir):
    df = load_data(input_path)
    print(f"Loaded {len(df)} rows for training")

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    df = df.dropna(subset=feature_cols, how="any")
    print(f"{len(df)} rows remain after dropping rows with missing feature values")

    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    y_log = np.log1p(y)

    # Fix category levels on the FULL dataset before splitting -- otherwise
    # a rare category that only ends up in the test split (e.g. a make with
    # just 1-2 listings) is "unseen" to XGBoost at predict time and it
    # raises an error. Locking categories in first avoids that.
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")

    X_train, X_test, y_train_log, y_test_log, y_train, y_test = train_test_split(
        X, y_log, y, test_size=0.2, random_state=42
    )

    train_baseline(X_train.copy(), X_test.copy(), y_train_log, y_test)
    model, metrics, importances = train_xgboost(X_train.copy(), X_test.copy(), y_train_log, y_test)

    # --- Save model + metadata ---
    import os
    os.makedirs(model_dir, exist_ok=True)
    model.save_model(os.path.join(model_dir, "xgb_model.json"))

    metadata = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "target_transform": "log1p",
        "test_metrics": metrics,
        "top_feature_importances": importances.head(10).to_dict(),
        # Store category levels so the API can validate/align inputs at inference time
        "category_levels": {col: sorted(X[col].dropna().unique().tolist()) for col in CATEGORICAL_FEATURES},
    }
    with open(os.path.join(model_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nSaved model to {model_dir}/xgb_model.json")
    print(f"Saved metadata to {model_dir}/model_metadata.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the car price prediction model")
    ap.add_argument("--input", type=str, default="../data/processed/cars_features.csv")
    ap.add_argument("--model-dir", type=str, default="../models")
    args = ap.parse_args()
    main(args.input, args.model_dir)
