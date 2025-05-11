import os
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
import hopsworks

# ---------------- CONFIG ----------------
TOP_STATIONS = ["JC115", "HB102", "HB103"]
METRICS_PATH = "data/metrics"
N_LAGS = 28
FUTURE_PERIODS = 168

# ---------------- SETUP ----------------
os.makedirs(METRICS_PATH, exist_ok=True)

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

# Load feature data
fg = fs.get_feature_group("citibike_features_dataset", version=1)
df = fg.read()

# ---------------- HELPERS ----------------
def create_lag_features(df, n_lags):
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["rides"].shift(lag)
    return df

# ---------------- MAIN ----------------
for station_id in TOP_STATIONS:
    print(f"🔮 Forecasting next {FUTURE_PERIODS} hours for {station_id} (Lag-28 model)...")

    station_df = df[df["start_station_id"] == station_id].sort_values("hour").copy()
    station_df = create_lag_features(station_df, N_LAGS).dropna()

    latest = station_df.iloc[-N_LAGS:].copy()

    # Load model from registry
    model_name = f"citibike_lag28_{station_id}"
    model_obj = mr.get_model(model_name, version=None)
    model_dir = model_obj.download()
    model = joblib.load(os.path.join(model_dir, "model.pkl"))

    predictions = []
    last_timestamp = latest["hour"].max()

    for _ in range(FUTURE_PERIODS):
        last_timestamp += timedelta(hours=1)
        last_lags = latest.tail(N_LAGS)["rides"].values[::-1]
        input_df = pd.DataFrame([last_lags], columns=[f"lag_{i}" for i in range(1, N_LAGS + 1)])
        pred = model.predict(input_df)[0]

        predictions.append({
            "hour": last_timestamp,
            "predicted_rides": pred
        })

        latest = pd.concat([latest, pd.DataFrame([{"hour": last_timestamp, "rides": pred}])], ignore_index=True)

    out_df = pd.DataFrame(predictions)
    out_file = f"{METRICS_PATH}/future_lgbm_lag28_{station_id}.csv"
    out_df.to_csv(out_file, index=False)
    print(f"✅ Saved: {out_file}")
