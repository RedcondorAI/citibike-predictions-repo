import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.mlflow_logger import set_mlflow_tracking, log_model_to_mlflow

import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
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

TOP_STATIONS = ["JC115", "HB102", "HB103"]
RESULTS_DIR = "data/metrics"
MODELS_DIR = "trained_models"
EXPERIMENT_NAME = "citibike-lgbm-pca"
MODEL_NAME = "LGBMPCA"
N_LAGS = 28
N_COMPONENTS = 10
STRATEGY = "lgbm_pca_reduction"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def create_lag_features(df, n_lags=28):
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["rides"].shift(lag)
    return df

def train_pca_model(df, station_id):
    station_df = df[df["start_station_id"] == station_id].copy()
    station_df = station_df.sort_values("hour")
    station_df = create_lag_features(station_df, N_LAGS).dropna()

    lag_cols = [f"lag_{i}" for i in range(1, N_LAGS + 1)]
    X = station_df[lag_cols]
    y = station_df["rides"]

    split_idx = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    pca = PCA(n_components=N_COMPONENTS)
    X_train = pca.fit_transform(X_train_raw)
    X_test = pca.transform(X_test_raw)

    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    explained_variance = float(round(sum(pca.explained_variance_ratio_) * 100, 2))

    model_dir = f"{MODELS_DIR}/pca_model_{station_id}"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/lgbm_model.pkl")
    joblib.dump(pca, f"{model_dir}/pca_transformer.pkl")

    # ✅ FIXED: Convert all values to standard types
    log_model_to_mlflow(
        model=model,
        input_data=X_test,
        experiment_name=EXPERIMENT_NAME,
        metric_name="mae",
        model_name=f"{MODEL_NAME}_{station_id}",
        score=float(mae),
        params={
            "station_id": station_id,
            "strategy": STRATEGY,
            "n_components": int(N_COMPONENTS),
            "explained_variance": explained_variance
        }
    )

    pd.DataFrame({
        "hour": station_df.iloc[split_idx:]["hour"],
        "actual_rides": y_test,
        "predicted_rides": y_pred
    }).to_csv(f"{RESULTS_DIR}/predictions_lgbm_pca_{station_id}.csv", index=False)

    return mae, explained_variance

def main():
    results = []
    for station_id in TOP_STATIONS:
        mae, variance = train_pca_model(df, station_id)
        results.append({
            "station_id": station_id,
            "model": MODEL_NAME,
            "strategy": STRATEGY,
            "mae": float(mae),
            "explained_variance": float(variance)
        })
    pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/lgbm_pca_mae_summary.csv", index=False)

if __name__ == "__main__":
    main()
