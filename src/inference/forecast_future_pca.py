import os
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from sklearn.decomposition import PCA
import hopsworks

# ---------------- CONFIG ----------------
TOP_STATIONS = ["JC115", "HB102", "HB103"]
METRICS_PATH = "data/metrics"
MODELS_PATH = "trained_models"
N_LAGS = 28
FUTURE_PERIODS = 168  # 7 days ahead
PCA_COMPONENTS = 10

# ---------------- SETUP ----------------
os.makedirs(METRICS_PATH, exist_ok=True)

# Connect to Hopsworks and load feature group
project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("citibike_features_dataset", version=1)
df = fg.read()

# ---------------- HELPERS ----------------
def create_lag_features(df, n_lags):
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["rides"].shift(lag)
    return df

# ---------------- MAIN ----------------
for station_id in TOP_STATIONS:
    print(f"🔮 Forecasting next {FUTURE_PERIODS} hours for {station_id} (PCA model)...")

    station_df = df[df["start_station_id"] == station_id].sort_values("hour").copy()
    station_df = create_lag_features(station_df, N_LAGS).dropna()

    latest = station_df.iloc[-N_LAGS:].copy()
    model_path = f"{MODELS_PATH}/lgbm_pca_model_{station_id}.pkl"

    if not os.path.exists(model_path):
        print(f"❌ Model not found for {station_id}: {model_path}")
        continue

    model = joblib.load(model_path)

    X_full = station_df[[f"lag_{i}" for i in range(1, N_LAGS + 1)]]
    pca = PCA(n_components=PCA_COMPONENTS)
    pca.fit(X_full)

    predictions = []
    last_timestamp = latest["hour"].max()

    for _ in range(FUTURE_PERIODS):
        last_timestamp += timedelta(hours=1)
        last_lags = latest.tail(N_LAGS)["rides"].values[::-1]
        input_df = pd.DataFrame([last_lags], columns=[f"lag_{i}" for i in range(1, N_LAGS + 1)])
        input_pca = pca.transform(input_df)

        pred = model.predict(input_pca)[0]

        predictions.append({
            "hour": last_timestamp,
            "predicted_rides": pred
        })

        new_row = {"hour": last_timestamp, "rides": pred}
        latest = pd.concat([latest, pd.DataFrame([new_row])], ignore_index=True)

    out_df = pd.DataFrame(predictions)
    out_file = f"{METRICS_PATH}/future_lgbm_pca_{station_id}.csv"
    out_df.to_csv(out_file, index=False)
    print(f"✅ Saved: {out_file}")
