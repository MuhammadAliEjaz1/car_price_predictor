"""
Hyperparameter tuning + error analysis for the car price predictor.

Input:  data/processed/cars_features.csv
Output: models/xgb_model.json, models/model_metadata.json (overwritten
        with the tuned model, if it beats the original)
        models/error_analysis.json (breakdown of where the model is weak)

This does two things:
  1. Randomized hyperparameter search (cross-validated) to see if the
     original train.py settings were leaving performance on the table.
  2. Error analysis on the held-out test set -- which makes/price ranges
     the model predicts well vs. poorly. This is the part that turns
     "I trained a model" into "I know its limitations," which matters
     more for a portfolio story than the metric itself.

Usage:
    python tune_and_analyze.py --input ../data/processed/cars_features.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split

NUMERIC_FEATURES = ["car_age", "mileage_km", "mileage_per_year", "engine_cc", "is_imported"]
CATEGORICAL_FEATURES = [
    "make", "model", "fuel_type", "transmission", "city",
    "assembly", "registered_in", "seller_type",
    "listing_province", "registered_province", "registered_matches_listing",
]
TARGET = "price_pkr"

PARAM_DISTRIBUTIONS = {
    "n_estimators": [200, 300, 400, 600, 800],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "reg_alpha": [0, 0.1, 0.5, 1.0],
    "reg_lambda": [1.0, 1.5, 2.0, 3.0],
}


def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=[TARGET])
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


def run_search(X_train, y_train_log, n_iter=40):
    base_model = xgb.XGBRegressor(enable_categorical=True, random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    print(f"Running randomized search: {n_iter} candidate configs x 5-fold CV...")
    search.fit(X_train, y_train_log)
    print(f"\nBest CV MAE (log scale): {-search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


def error_analysis(model, X_test, y_test, df_test_meta):
    """
    Break down prediction error by make and by price bucket, on the
    held-out test set. df_test_meta carries make/model/price for
    readable reporting (X_test itself has categorical dtypes already
    encoded for the model).
    """
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)

    results = df_test_meta.copy()
    results["predicted_price"] = preds
    results["actual_price"] = y_test.values
    results["abs_error"] = (results["predicted_price"] - results["actual_price"]).abs()
    results["pct_error"] = results["abs_error"] / results["actual_price"] * 100

    # --- Error by make (only makes with enough test-set listings to be meaningful) ---
    by_make = (
        results.groupby("make")
        .agg(count=("pct_error", "size"), mean_mape=("pct_error", "mean"), median_mape=("pct_error", "median"))
        .query("count >= 5")
        .sort_values("mean_mape", ascending=False)
    )
    print("\n--- Worst-predicted makes (min 5 test listings) ---")
    print(by_make.head(10).to_string())
    print("\n--- Best-predicted makes (min 5 test listings) ---")
    print(by_make.tail(10).to_string())

    # --- Error by price bucket ---
    results["price_bucket"] = pd.qcut(
        results["actual_price"], q=4, labels=["Low (Q1)", "Mid-low (Q2)", "Mid-high (Q3)", "High (Q4)"]
    )
    by_bucket = results.groupby("price_bucket", observed=True).agg(
        count=("pct_error", "size"), mean_mape=("pct_error", "mean"), mean_abs_error=("abs_error", "mean")
    )
    print("\n--- Error by price quartile ---")
    print(by_bucket.to_string())

    return {
        "worst_makes": by_make.head(10).reset_index().to_dict(orient="records"),
        "best_makes": by_make.tail(10).reset_index().to_dict(orient="records"),
        "by_price_quartile": by_bucket.reset_index().astype(str).to_dict(orient="records"),
    }


def main(input_path, model_dir, n_iter):
    df = load_data(input_path)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    df = df.dropna(subset=feature_cols, how="any")
    print(f"{len(df)} rows available for tuning")

    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    y_log = np.log1p(y)

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")

    X_train, X_test, y_train_log, y_test_log, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y_log, y, df[["make", "model"]], test_size=0.2, random_state=42
    )

    # --- Baseline: original train.py default params, for comparison ---
    default_model = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        enable_categorical=True, random_state=42, n_jobs=-1,
    )
    default_model.fit(X_train, y_train_log)
    default_preds = np.expm1(default_model.predict(X_test))
    original_metrics = evaluate(y_test, default_preds, "Original (train.py defaults)")

    # --- Tuned model ---
    best_model, best_params = run_search(X_train, y_train_log, n_iter=n_iter)
    tuned_preds = np.expm1(best_model.predict(X_test))
    tuned_metrics = evaluate(y_test, tuned_preds, "Tuned (RandomizedSearchCV)")

    improved = tuned_metrics["mae"] < original_metrics["mae"]
    if improved:
        print("\nTuned model improves on original.")
    else:
        print("\nTuning did not beat the original -- keeping original.")

    final_model = best_model if improved else default_model
    final_metrics = tuned_metrics if improved else original_metrics

    # --- Error analysis on whichever model we're keeping ---
    error_report = error_analysis(final_model, X_test, y_test, meta_test)

    # --- Save ---
    os.makedirs(model_dir, exist_ok=True)
    final_model.save_model(os.path.join(model_dir, "xgb_model.json"))

    metadata_path = os.path.join(model_dir, "model_metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    metadata["test_metrics"] = final_metrics
    metadata["tuning"] = {
        "attempted": True,
        "improved_on_original": improved,
        "original_metrics": original_metrics,
        "best_params": best_params if improved else "not used (original kept)",
    }
    importances = pd.Series(final_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    metadata["top_feature_importances"] = importances.head(10).to_dict()
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    with open(os.path.join(model_dir, "error_analysis.json"), "w") as f:
        json.dump(error_report, f, indent=2, default=str)

    print(f"\nSaved model to {model_dir}/xgb_model.json")
    print(f"Saved metadata to {metadata_path}")
    print(f"Saved error analysis to {model_dir}/error_analysis.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tune the model and analyze prediction errors")
    ap.add_argument("--input", type=str, default="../data/processed/cars_features.csv")
    ap.add_argument("--model-dir", type=str, default="../models")
    ap.add_argument("--n-iter", type=int, default=40, help="number of random search candidates to try")
    args = ap.parse_args()
    main(args.input, args.model_dir, args.n_iter)
