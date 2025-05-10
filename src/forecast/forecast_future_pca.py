import os
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

# Constants
DATA_PATH = "data/processed/jc_hourly_features.csv"
MODEL_DIR = "trained_models"
OUTPUT_DIR = "data/metrics"
FORECAST_HORIZON = 168  # 7 days
N_LAGS = 28
TOP_STATIONS = ["JC115", "HB102", "HB103"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_lag_feature_df(lag_values):
    """Creates a DataFrame of shape (1, N_LAGS) with lag_1 to lag_28."""
    return pd.DataFrame({f"lag_{i+1}": [lag_values[-(i+1)]] for i in range(N_LAGS)})

def forecast_pca_model(station_id):
    # Load historical data
    df = pd.read_csv(DATA_PATH, parse_dates=["hour"])
    df = df[df["start_station_id"] == station_id].sort_values("hour")
    lag_series = df["rides"].values[-N_LAGS:]

    # Paths to model and PCA
    model_dir = os.path.join(MODEL_DIR, f"pca_model_{station_id}")
    model_path = os.path.join(model_dir, "lgbm_model.pkl")
    pca_path = os.path.join(model_dir, "pca_transformer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(pca_path):
        print(f"❌ Missing PCA model or transformer for {station_id}")
        return

    model = joblib.load(model_path)
    pca = joblib.load(pca_path)

    # Forecast
    last_time = df["hour"].max()
    predictions, timestamps = [], []
    recent = lag_series.copy()

    for i in range(FORECAST_HORIZON):
        X_input = create_lag_feature_df(recent)
        X_pca = pca.transform(X_input)

        pred = model.predict(X_pca)[0]
        future_time = last_time + timedelta(hours=i + 1)

        timestamps.append(future_time)
        predictions.append(pred)
        recent = np.append(recent[1:], pred)

    # Save forecast
    forecast_df = pd.DataFrame({
        "hour": timestamps,
        "predicted_rides": predictions
    })
    out_path = os.path.join(OUTPUT_DIR, f"future_lgbm_pca_{station_id}.csv")
    forecast_df.to_csv(out_path, index=False)
    print(f"✅ Saved PCA forecast for {station_id} to {out_path}")

def main():
    for station_id in TOP_STATIONS:
        forecast_pca_model(station_id)

if __name__ == "__main__":
    main()
