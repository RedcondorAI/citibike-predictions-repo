import os
import pandas as pd
import joblib
import hopsworks
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = "data/metrics"
MODELS_DIR = "trained_models"
PREDICTION_STORE_NAME = "citibike_predictions_dataset"
TOP_STATIONS = ["JC115", "HB102", "HB103"]

def upload_naive_predictions_and_models(project):
    fs = project.get_feature_store()
    registry = project.get_model_registry()

    mae_df = pd.read_csv(os.path.join(RESULTS_DIR, "baseline_mae_summary.csv"))

    for station_id in TOP_STATIONS:
        # 1. Upload Predictions
        pred_path = os.path.join(RESULTS_DIR, f"predictions_naive_{station_id}.csv")
        if not os.path.exists(pred_path):
            print(f"❌ Missing predictions: {pred_path}")
            continue

        df = pd.read_csv(pred_path, parse_dates=["hour"])
        fg_name = f"{PREDICTION_STORE_NAME}_naive_{station_id.lower()}"
        fg = fs.get_or_create_feature_group(
            name=fg_name,
            version=1,
            primary_key=["hour"],
            event_time="hour"
        )
        fg.insert(df)
        print(f"✅ Uploaded naive predictions → {fg_name}")

        # 2. Register Model
        model_path = os.path.join(MODELS_DIR, f"naive_model_{station_id}.pkl")
        if not os.path.exists(model_path):
            print(f"❌ Missing model: {model_path}")
            continue

        model = joblib.load(model_path)
        tmp_dir = f"tmp_model_naive_{station_id}"
        os.makedirs(tmp_dir, exist_ok=True)
        joblib.dump(model, os.path.join(tmp_dir, "model.pkl"))

        mae = 0
        try:
            mae = float(mae_df.loc[mae_df["station_id"] == station_id, "mae"].values[0])
        except Exception:
            print(f"⚠️ Could not extract MAE for {station_id}")

        X_dummy = pd.DataFrame({"predicted_rides": [0.0]})
        y_dummy = pd.Series([0.0])

        model_obj = registry.python.create_model(
            name=f"citibike_naive_{station_id}",
            metrics={"mae": mae},
            model_schema=ModelSchema(Schema(X_dummy), Schema(y_dummy)),
            description=f"Naive model (lag-1) for {station_id}"
        )
        model_obj.save(tmp_dir)
        print(f"✅ Registered model: citibike_naive_{station_id}")

        import shutil
        shutil.rmtree(tmp_dir)

def main():
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT_NAME"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    upload_naive_predictions_and_models(project)

if __name__ == "__main__":
    main()
