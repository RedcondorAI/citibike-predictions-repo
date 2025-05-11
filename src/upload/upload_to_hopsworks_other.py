import pandas as pd
import hopsworks
import os

# Login to Hopsworks
project = hopsworks.login(
    project=os.environ["HOPSWORKS_PROJECT_NAME"],
    api_key_value=os.environ["HOPSWORKS_API_KEY"]
)

fs = project.get_feature_store()

def upload_metrics(file_name, fg_name, version=1):
    print(f"Uploading {file_name} to feature group: {fg_name}")
    df = pd.read_csv(file_name)

    fg = fs.get_or_create_feature_group(
        name=fg_name,
        version=version,
        description=f"{fg_name} MAE summary",
        primary_key=["station_id"],
        event_time=None,
        online_enabled=False  # ✅ disable Kafka
    )

    fg.insert(df, write_options={"wait_for_job": True, "write_mode": "overwrite"})
    print(f"✅ Uploaded to {fg_name}")

# Upload both model summaries
upload_metrics("data/metrics/baseline_mae_summary.csv", "citibike_model_metrics_baseline")
upload_metrics("data/metrics/lgbm_topk_mae_summary.csv", "citibike_model_metrics_topk")
