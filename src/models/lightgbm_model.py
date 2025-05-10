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
EXPERIMENT_NAME = "citibike-lgbm-lag28"
MODEL_NAME = "LGBMLag28"
N_LAGS = 28
STRATEGY = "lgbm_all_lags"  # Added strategy name

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
mlflow = set_mlflow_tracking()


def create_lag_features(df, n_lags=28):
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["rides"].shift(lag)
    return df


def train_and_log_model(df, station_id):
    station_df = df[df["start_station_id"] == station_id].copy()
    station_df = station_df.sort_values("hour")
    station_df = create_lag_features(station_df, N_LAGS)
    station_df = station_df.dropna()

    feature_cols = [f"lag_{i}" for i in range(1, N_LAGS + 1)]
    X = station_df[feature_cols]
    y = station_df["rides"]

    # Simple train/test split (last 20% as test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # Save the model to disk using joblib
    model_path = f"{MODELS_DIR}/lgbm_lag28_model_{station_id}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Log to MLflow with detailed strategy parameter
    log_model_to_mlflow(
        model=model,
        input_data=X_test,
        experiment_name=EXPERIMENT_NAME,
        metric_name="mae",
        model_name=f"{MODEL_NAME}_{station_id}",
        params={
            "n_lags": N_LAGS, 
            "station_id": station_id,
            "strategy": STRATEGY,  # Added strategy parameter
            "model_type": "lgbm",  # Added model type
            "feature_count": len(feature_cols)  # Added feature count
        },
        score=mae
    )

    # Save predictions
    preds_df = pd.DataFrame({
        "hour": station_df.iloc[split_idx:]["hour"].values,
        "actual_rides": y_test.values,
        "predicted_rides": y_pred
    })
    preds_df.to_csv(f"{RESULTS_DIR}/predictions_lgbm_lag28_{station_id}.csv", index=False)

    return mae


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["hour"])
    results = []

    for station_id in TOP_STATIONS:
        mae = train_and_log_model(df, station_id)
        results.append({
            "station_id": station_id, 
            "model": MODEL_NAME, 
            "strategy": STRATEGY,  # Added strategy to results
            "mae": mae
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{RESULTS_DIR}/lgbm_lag28_mae_summary.csv", index=False)
    print(f"\nLightGBM {STRATEGY} MAE Results:")
    print(results_df)


if __name__ == "__main__":
    main()