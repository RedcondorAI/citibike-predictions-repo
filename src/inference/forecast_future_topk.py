import os
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

# Constants
DATA_PATH = "data/processed/jc_hourly_features.csv"
MODEL_DIR = "trained_models"
OUTPUT_DIR = "data/metrics"
FORECAST_HORIZON = 168  # 7 days hourly
TOP_STATIONS = ["JC115", "HB102", "HB103"]
N_LAGS = 28
TOP_K = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_lag_features(series, topk_cols):
    """Build a feature DataFrame with top-K lag columns using the latest values."""
    return pd.DataFrame({col: [series[-int(col.split('_')[1])]] for col in topk_cols})

def forecast_future_rides_topk(station_id):
    # Load data
    df = pd.read_csv(DATA_PATH, parse_dates=["hour"])
    df = df[df["start_station_id"] == station_id].sort_values("hour")
    lag_series = df["rides"].values[-N_LAGS:]

    # Load model
    model_path = os.path.join(MODEL_DIR, f"lgbm_topk_model_{station_id}.pkl")
    if not os.path.exists(model_path):
        print(f"❌ Model not found for {station_id}")
        return

    model = joblib.load(model_path)

    # Get top-K columns from model booster if available
    booster = model.booster_
    importance_df = pd.DataFrame({
        "feature": booster.feature_name(),
        "importance": booster.feature_importance()
    }).sort_values("importance", ascending=False)
    topk_cols = importance_df.head(TOP_K)["feature"].tolist()

    # Forecast
    last_time = df["hour"].max()
    predictions, timestamps = [], []
    recent = lag_series.copy()

    for i in range(FORECAST_HORIZON):
        X_input = create_lag_features(recent, topk_cols)
        pred = model.predict(X_input)[0]
        future_time = last_time + timedelta(hours=i + 1)

        timestamps.append(future_time)
        predictions.append(pred)
        recent = np.append(recent[1:], pred)

    # Save output
    out_path = os.path.join(OUTPUT_DIR, f"future_lgbm_topk_{station_id}.csv")
    forecast_df = pd.DataFrame({
        "hour": timestamps,
        "predicted_rides": predictions
    })
    forecast_df.to_csv(out_path, index=False)
    print(f"✅ Forecast saved: {out_path}")

def main():
    for station in TOP_STATIONS:
        forecast_future_rides_topk(station)

if __name__ == "__main__":
    main()
