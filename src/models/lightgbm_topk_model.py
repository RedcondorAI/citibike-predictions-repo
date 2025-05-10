import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from mlflow_logger import set_mlflow_tracking, log_model_to_mlflow

DATA_PATH = "data/processed/jc_hourly_features.csv"
TOP_STATIONS = ["JC115", "HB102", "HB103"]
RESULTS_DIR = "data/metrics"
MODELS_DIR = "trained_models"
EXPERIMENT_NAME = "citibike-lgbm-topk"
MODEL_NAME = "LGBMTopK"
N_LAGS = 28
TOP_K = 10  # number of top features to select

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
mlflow = set_mlflow_tracking()


def create_lag_features(df, n_lags=28):
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["rides"].shift(lag)
    return df


def train_topk_model(df, station_id):
    station_df = df[df["start_station_id"] == station_id].copy()
    station_df = station_df.sort_values("hour")
    station_df = create_lag_features(station_df, N_LAGS)
    station_df = station_df.dropna()

    lag_cols = [f"lag_{i}" for i in range(1, N_LAGS + 1)]
    X_full = station_df[lag_cols]
    y = station_df["rides"]

    split_idx = int(len(X_full) * 0.8)
    X_train_full, X_test_full = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # First model to get feature importance
    lgb_full = LGBMRegressor(random_state=42)
    lgb_full.fit(X_train_full, y_train)

    # Get top K features
    importances = lgb_full.feature_importances_
    topk_idx = np.argsort(importances)[::-1][:TOP_K]
    topk_features = [lag_cols[i] for i in topk_idx]

    # Train new model with top-K features
    X_train_topk = X_train_full[topk_features]
    X_test_topk = X_test_full[topk_features]

    model_topk = LGBMRegressor(random_state=42)
    model_topk.fit(X_train_topk, y_train)

    y_pred = model_topk.predict(X_test_topk)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Save the model to disk using joblib
    model_path = f"{MODELS_DIR}/lgbm_topk_model_{station_id}.pkl"
    joblib.dump(model_topk, model_path)
    print(f"Model saved to {model_path}")

    # Log to MLflow
    log_model_to_mlflow(
        model=model_topk,
        input_data=X_test_topk,
        experiment_name=EXPERIMENT_NAME,
        metric_name="mae",
        model_name=f"{MODEL_NAME}_{station_id}",
        params={"top_k": TOP_K, "station_id": station_id},
        score=mae
    )

    # Save predictions
    preds_df = pd.DataFrame({
        "hour": station_df.iloc[split_idx:]["hour"].values,
        "actual_rides": y_test.values,
        "predicted_rides": y_pred
    })
    preds_df.to_csv(f"{RESULTS_DIR}/predictions_lgbm_topk_{station_id}.csv", index=False)

    return mae


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["hour"])
    results = []

    for station_id in TOP_STATIONS:
        mae = train_topk_model(df, station_id)
        results.append({"station_id": station_id, "model": MODEL_NAME, "mae": mae})

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{RESULTS_DIR}/lgbm_topk_mae_summary.csv", index=False)
    print("\nLightGBM Top-K MAE Results:")
    print(results_df)


if __name__ == "__main__":
    main()
