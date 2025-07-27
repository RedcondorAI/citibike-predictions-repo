#force rerun1
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.mlflow_logger import set_mlflow_tracking, log_model_to_mlflow

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor
import mlflow
import joblib
import hopsworks

# Initialize MLflow
set_mlflow_tracking()

# Load feature data from Hopsworks
project = hopsworks.login(
    project=os.environ["HOPSWORKS_PROJECT_NAME"],
    api_key_value=os.environ["HOPSWORKS_API_KEY"]
)
fs = project.get_feature_store()
fg = fs.get_feature_group("citibike_features_dataset", version=1)
df = fg.read()

# Model settings
TOP_STATIONS = ["JC115", "HB102", "HB103"]
RESULTS_DIR = "data/metrics"
MODELS_DIR = "trained_models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def run_naive_baseline(df, station_id):
    """Run a naive lag-1 baseline model using DummyRegressor for a single station."""
    station_df = df[df["start_station_id"] == station_id].copy()
    station_df = station_df.sort_values("hour")
    station_df["predicted_rides"] = station_df["rides"].shift(1)
    station_df = station_df.dropna(subset=["predicted_rides"])

    X = station_df[["predicted_rides"]]
    y = station_df["rides"]

    dummy_model = DummyRegressor(strategy="mean")
    dummy_model.fit(X, y)

    mae = mean_absolute_error(y, X["predicted_rides"])
    return mae, station_df, dummy_model, X

def main():
    results = []

    for station_id in TOP_STATIONS:
        mae, preds, model, X_train = run_naive_baseline(df, station_id)

        results.append({
            "station_id": station_id,
            "model": "naive_lag_1",
            "mae": mae
        })

        # Save predictions
        pred_path = f"{RESULTS_DIR}/predictions_naive_{station_id}.csv"
        preds.to_csv(pred_path, index=False)

        # Save model
        model_path = f"{MODELS_DIR}/naive_model_{station_id}.pkl"
        joblib.dump(model, model_path)

        # Log model to MLflow
        log_model_to_mlflow(
            model=model,
            input_data=X_train,
            experiment_name="citibike-baseline",
            metric_name="mae",
            model_name=None,        #"BaselineModelNaiveLag"
            score=mae,
            params={"station_id": station_id, "strategy": "naive_lag_1"},
        )

    # Save summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{RESULTS_DIR}/baseline_mae_summary.csv", index=False)

    print("\nBaseline MAE Results:")
    print(results_df)

if __name__ == "__main__":
    main()
