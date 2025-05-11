import pandas as pd
import hopsworks
import os

# Login to Hopsworks
project = hopsworks.login(
    project=os.environ["HOPSWORKS_PROJECT_NAME"],
    api_key_value=os.environ["HOPSWORKS_API_KEY"]
)
fs = project.get_feature_store()

df = pd.read_csv("data/metrics/lgbm_pca_mae_summary.csv")

fg = fs.get_or_create_feature_group(
    name="citibike_model_metrics_pca",
    version=1,
    description="MAE and explained variance for PCA model",
    primary_key=["station_id"],
    event_time=None,
    online_enabled=False,
    storage_connector="Offline"
)

fg.insert(df, write_options={"wait_for_job": True, "write_mode": "overwrite"})
print("✅ PCA metrics uploaded to Hopsworks.")
