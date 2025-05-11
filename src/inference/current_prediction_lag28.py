import os
import pandas as pd
import numpy as np
import joblib
import hopsworks

# ---------------- CONFIG ----------------
TOP_STATIONS = ["JC115", "HB102", "HB103"]
METRICS_PATH = "data/metrics"
MODELS_PATH = "trained_models"
N_LAGS = 28

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
for station in TOP_STATIONS:
    print(f"📈 Generating current test predictions for {station} using Lag-28 model...")

    station_df = df[df["start_station_id"] == station].sort_values("hour").copy()
    station_df = create_lag_features(station_df, N_LAGS).dropna()

    X = station_df[[f"lag_{i}" for i in range(1, N_LAGS + 1)]]
    y = station_df["rides"]
    split = int(len(X) * 0.8)

    X_test, y_test = X.iloc[split:], y.iloc[split:]
    model_path = f"{MODELS_PATH}/lgbm_lag28_model_{station}.pkl"

    if not os.path.exists(model_path):
        print(f"❌ Model not found for {station}: {model_path}")
        continue

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    out_df = pd.DataFrame({
        "hour": station_df.iloc[split:]["hour"],
        "actual_rides": y_test,
        "predicted_rides": y_pred
    })

    out_file = f"{METRICS_PATH}/predictions_lgbm_lag28_{station}.csv"
    out_df.to_csv(out_file, index=False)
    print(f"✅ Saved: {out_file}")
