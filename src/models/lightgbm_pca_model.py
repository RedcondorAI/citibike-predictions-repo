import pandas as pd
import numpy as np
import os
import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from mlflow_logger import set_mlflow_tracking, log_model_to_mlflow

DATA_PATH = "data/processed/jc_hourly_features.csv"
TOP_STATIONS = ["JC115", "HB102", "HB103"]
RESULTS_DIR = "data/metrics"
MODELS_DIR = "trained_models"  # Added models directory
EXPERIMENT_NAME = "citibike-lgbm-pca"
MODEL_NAME = "LGBMPCA"
N_LAGS = 28
N_COMPONENTS = 10  # can also use variance threshold if preferred
STRATEGY = "lgbm_pca_reduction"  # Added strategy name

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)  # Create models directory if it doesn't exist
mlflow = set_mlflow_tracking()


def create_lag_features(df, n_lags=28):
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["rides"].shift(lag)
    return df


def train_pca_model(df, station_id):
    station_df = df[df["start_station_id"] == station_id].copy()
    station_df = station_df.sort_values("hour")
    station_df = create_lag_features(station_df, N_LAGS)
    station_df = station_df.dropna()

    lag_cols = [f"lag_{i}" for i in range(1, N_LAGS + 1)]
    X = station_df[lag_cols]
    y = station_df["rides"]

    split_idx = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Fit PCA on training data only
    pca = PCA(n_components=N_COMPONENTS)
    X_train_pca = pca.fit_transform(X_train_raw)
    X_test_pca = pca.transform(X_test_raw)
    
    # Calculate explained variance
    explained_variance = sum(pca.explained_variance_ratio_) * 100
    
    model = LGBMRegressor(random_state=42)
    model.fit(X_train_pca, y_train)

    y_pred = model.predict(X_test_pca)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Save both PCA and model
    model_dir = f"{MODELS_DIR}/pca_model_{station_id}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the LightGBM model
    model_path = f"{model_dir}/lgbm_model.pkl"
    joblib.dump(model, model_path)
    
    # Save the PCA transformer
    pca_path = f"{model_dir}/pca_transformer.pkl"
    joblib.dump(pca, pca_path)
    
    print(f"Model and PCA transformer saved to {model_dir}")

    # Log to MLflow with detailed parameters
    log_model_to_mlflow(
        model=model,
        input_data=X_test_pca,
        experiment_name=EXPERIMENT_NAME,
        metric_name="mae",
        model_name=f"{MODEL_NAME}_{station_id}",
        params={
            "n_components": N_COMPONENTS, 
            "station_id": station_id,
            "strategy": STRATEGY,  # Added strategy parameter
            "model_type": "lgbm",  # Added model type
            "explained_variance": round(explained_variance, 2),  # Added explained variance
            "original_features": len(lag_cols)  # Added original feature count
        },
        score=mae
    )

    preds_df = pd.DataFrame({
        "hour": station_df.iloc[split_idx:]["hour"].values,
        "actual_rides": y_test.values,
        "predicted_rides": y_pred
    })
    preds_df.to_csv(f"{RESULTS_DIR}/predictions_lgbm_pca_{station_id}.csv", index=False)

    return mae, explained_variance


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["hour"])
    results = []

    for station_id in TOP_STATIONS:
        mae, explained_variance = train_pca_model(df, station_id)
        results.append({
            "station_id": station_id, 
            "model": MODEL_NAME, 
            "strategy": STRATEGY,  # Added strategy to results
            "mae": mae,
            "explained_variance": round(explained_variance, 2)  # Track explained variance
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{RESULTS_DIR}/lgbm_pca_mae_summary.csv", index=False)
    print(f"\nLightGBM {STRATEGY} MAE Results:")
    print(results_df)


if __name__ == "__main__":
    main()