# import os
# from pathlib import Path
# from dotenv import load_dotenv

# # Load environment variables from .env
# load_dotenv()

# # Project root
# PROJECT_ROOT = Path(__file__).resolve().parent.parent

# # Data directories
# DATA_DIR = PROJECT_ROOT / "data"
# RAW_DATA_DIR = DATA_DIR / "raw"
# PROCESSED_DATA_DIR = DATA_DIR / "processed"
# METRICS_DIR = DATA_DIR / "metrics"

# # Model directory
# MODELS_DIR = PROJECT_ROOT / "trained_models"

# # Feature engineering output
# FEATURES_PATH = PROCESSED_DATA_DIR / "jc_hourly_features.csv"

# # MLflow and Hopsworks credentials
# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
# HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
# HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

# # Top 3 station IDs (based on your earlier analysis)
# TOP_STATIONS = ["JC115", "HB102", "HB103"]

# # Hopsworks Feature Store definitions
# FEATURE_GROUP_NAME = "citibike_features_dataset"
# PREDICTION_GROUP_NAME = "citibike_predictions_dataset"

# # MAE summary file
# LGBM_MAE_CSV = METRICS_DIR / "lgbm_lag28_mae_summary.csv"
