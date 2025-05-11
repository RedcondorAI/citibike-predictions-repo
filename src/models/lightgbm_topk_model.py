import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.mlflow_logger import set_mlflow_tracking, log_model_to_mlflow

import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import hopsworks

# Setup MLflow tracking
set_mlflow_tracking()

# Connect to Hopsworks
project = hopsworks.login(
    project=os.environ["HOPSWORKS_PROJECT_NAME"],
    api_key_value=os.environ["HOPSWORKS_API_KEY"]
)
fs = project.get_feature_store()
fg = fs.get_feature_group("citibike_features_dataset", version=1)
df = fg.read()

# Settings
TOP_STATIONS = ["JC115", "HB102", "HB103"]
RESULTS_DIR = "data/metrics"
MODELS_DIR = "trained_models"
EXPERIMENT_NAME = "citibike-lgbm-topk"
MODEL_NAME = "LGBMTopK"
N_LAGS = 28
TOP_K = 10
STRATEGY = "lgbm_top_k"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def create_lag_features(df, n_lags=28):
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["rides"].shift(lag)
    return df

def train_and_log_model(df, station_id):
    station_df = df[df["start_station_id"] == station_id].copy()
    station_df = station_df.sort_values("hour")
    station_df = create_lag_features(station_df, N_LAGS).dropna()

    full_features = [f"lag_{i}" for i in range(1, N_LAGS + 1)]
    X = station_df[full_features]
    y = station_df["rides"]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # First model to get top-k important features
    temp_model = LGBMRegressor(random_state=42)
    temp_model.fit(X_train, y_train)
    importance = temp_model.feature_importances_
    top_k_idx = np.argsort(importance)[-TOP_K:]
    top_k_features = [full_features[i] for i in top_k_idx]

    # Second model trained on top-k features
    model = LGBMRegressor(random_state=42)
    model.fit(X_train[top_k_features], y_train)
    y_pred = model.predict(X_test[top_k_features])
    mae = mean_absolute_error(y_test, y_pred)

    joblib.dump(model, f"{MODELS_DIR}/lgbm_topk_model_{station_id}.pkl")

    # ✅ Fixed: pass all values correctly
    log_model_to_mlflow(
        model=model,
        input_data=X_test[top_k_features],
        experiment_name=EXPERIMENT_NAME,
        metric_name="mae",
        model_name=f"{MODEL_NAME}_{station_id}",
        score=float(mae),
        params={
            "station_id": station_id,
            "strategy": STRATEGY,
            "top_k": int(TOP_K)
        }
    )

    # Save predictions
    pd.DataFrame({
        "hour": station_df.iloc[split:]["hour"],
        "actual_rides": y_test,
        "predicted_rides": y_pred
    }).to_csv(f"{RESULTS_DIR}/lgbm_topk_{station_id}.csv", index=False)

    return mae

def main():
    results = []
    for station_id in TOP_STATIONS:
        mae = train_and_log_model(df, station_id)
        results.append({
            "station_id": station_id,
            "model": MODEL_NAME,
            "strategy": STRATEGY,
            "mae": float(mae)
        })

    pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/lgbm_topk_mae_summary.csv", index=False)

if __name__ == "__main__":
    main()
