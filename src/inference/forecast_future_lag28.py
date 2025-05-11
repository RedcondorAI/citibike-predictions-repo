import os
import pandas as pd
import joblib
from datetime import timedelta

# Constants
DATA_PATH = "data/processed/jc_hourly_features.csv"
MODEL_DIR = "trained_models"
OUTPUT_DIR = "data/metrics"
FORECAST_HORIZON = 168  # e.g., 7 days of hourly predictions
N_LAGS = 28
TOP_STATIONS = ["JC115", "HB102", "HB103"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_lag_features(series, n_lags):
    arr = series.to_numpy()
    data = {f"lag_{i+1}": [arr[-(i+1)]] for i in range(n_lags)}
    return pd.DataFrame(data)



def forecast_future_rides(station_id):
    # Load latest processed data
    df = pd.read_csv(DATA_PATH, parse_dates=["hour"])
    df = df[df["start_station_id"] == station_id].sort_values("hour")

    # Extract recent lags
    latest_series = df["rides"].values
    recent_lags = latest_series[-N_LAGS:]
    timestamps = []
    predictions = []

    # Load model
    model_path = os.path.join(MODEL_DIR, f"lgbm_lag28_model_{station_id}.pkl")
    if not os.path.exists(model_path):
        print(f"❌ Model not found for {station_id}")
        return
    model = joblib.load(model_path)

    # Generate future timestamps
    last_time = df["hour"].max()
    for i in range(FORECAST_HORIZON):
        input_df = create_lag_features(pd.Series(recent_lags), N_LAGS)
        pred = model.predict(input_df)[0]

        # Store prediction
        future_time = last_time + timedelta(hours=i + 1)
        timestamps.append(future_time)
        predictions.append(pred)

        # Update lags
        recent_lags = list(recent_lags[1:]) + [pred]

    # Save results
    forecast_df = pd.DataFrame({
        "hour": timestamps,
        "predicted_rides": predictions
    })
    forecast_df.to_csv(os.path.join(OUTPUT_DIR, f"future_lgbm_lag28_{station_id}.csv"), index=False)
    print(f"✅ Forecast saved for {station_id}")

def main():
    for station in TOP_STATIONS:
        forecast_future_rides(station)

if __name__ == "__main__":
    main()
