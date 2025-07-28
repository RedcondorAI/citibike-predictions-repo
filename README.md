# 🚲 Citi Bike Trip Prediction System

![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)
![License](https://img.shields.io/github.com/RedcondorAI/citibike-predictions-repo.git))
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit\&logoColor=white)
![Hopsworks](https://img.shields.io/badge/Feature%20Store-Hopsworks-20C997?logo=data\:image/svg+xml;base64,)

> End-to-end machine learning system for forecasting hourly Citi Bike demand in Jersey City 🚴‍♂️⚡

---

## 📜 Project Summary

A fully automated and modular time series forecasting pipeline built using:

* ✅ **Citi Bike historical data (2023–2025)**
* ⚙️ LightGBM models: full-lag, top-k, PCA-reduced
* ☁️ Cloud integration with **Hopsworks** and **MLflow**
* 📊 Streamlit apps for visualization and monitoring
* 🤖 CI automation with **GitHub Actions**

---

## 📅 Full End-to-End Pipeline

### 🔶 Feature Engineering

This pipeline is designed with a **two-phase setup**:

#### 🔹 Phase 1: One-time Historical Bootstrapping

Used to fetch and upload the initial 2+ years of Citi Bike data (2023–April 2025).

```bash
python src/data/fetch_data.py
python src/data/preprocess_data.py
python src/features/engineering_features.py
python src/upload/upload_features_to_hopsworks.py
```

This loads the historical backbone of the system into the Hopsworks Feature Store and is only needed **once**.

#### 🔸 Phase 2: Ongoing Automation via GitHub Actions

After the initial upload, all future data ingestion is handled incrementally:

```bash
python src/data/fetch_recent_data.py
python src/data/preprocess_recent_data.py
python src/features/engineer_recent_features.py
python src/upload/upload_recent_to_hopsworks.py
```

* ✅ Fetches only the last 2 months of data
* ✅ Ensures complete lag coverage for new timestamps
* ✅ Filters out already-uploaded data by comparing timestamps

This pattern lets the system safely **continue evolving without reprocessing the past**, and keeps Hopsworks clean and consistent.

---

### 🔵 Model Training

Triggered automatically after feature upload:

```bash
python src/models/baseline_model.py
python src/models/lightgbm_model.py
python src/models/lightgbm_topk_model.py
python src/models/lightgbm_pca_model.py
```

Uploads to Hopsworks:

```bash
python src/upload/upload_to_hopsworks_other.py
python src/upload/upload_to_hopsworks_best.py
python src/upload/upload_to_hopsworks_pca.py
```

### 🔸 Inference Pipeline

```bash
python src/inference/current_prediction_*.py
python src/inference/forecast_future_*.py
python src/upload/upload_to_hopsworks_inference.py
```

### 🔹 Streamlit Dashboards

```bash
streamlit run app/app.py        # Forecast app
streamlit run app/monitor_app.py  # Monitoring app
```

---

## 📊 System Architecture

1. ⬇️ Download Citi Bike data (last 2 months)
2. 💚 Engineer features (lag, time-based)
3. ⬆️ Upload only new hours to Hopsworks
4. 🤖 Train 4 models (baseline, lag-28, top-K, PCA)
5. 🔢 Log everything to MLflow + Hopsworks
6. 🌎 Forecast 7 days ahead and upload predictions
7. 📊 Visualize in real-time via Streamlit

---

## ⚙️ GitHub Actions Automation

| Workflow                  | Trigger                    | Description                         |
| ------------------------- | -------------------------- | ----------------------------------- |
| `feature_engineering.yml` | Manual / Cron (1st & 15th) | Downloads, cleans, uploads features |
| `train_models.yml`        | After feature engineering  | Trains and registers models         |
| `inference.yml`           | After model training       | Generates and uploads predictions   |

---

## 📁 Directory Structure

```
citibike-predictions/
├── data/                      # Raw, processed, and prediction data
├── src/
│   ├── data/                 # fetch, fetch_recent, preprocess, preprocess_recent
│   ├── features/             # engineering_features.py, engineer_recent_features.py
│   ├── models/               # All model training scripts
│   ├── inference/            # Prediction and forecasting scripts
│   ├── upload/               # All upload-to-Hopsworks scripts
├── app/                      # Streamlit dashboards
├── .github/workflows/        # GitHub Actions pipelines
├── requirements.txt
└── README.md
```

---

## 📃 Data Sources & Tools

* 🚲 [Citi Bike NYC Trip Data](https://citibikenyc.com/system-data)
* ☁️ [Hopsworks Feature Store](https://www.hopsworks.ai/)
* 🔬 [MLflow Logging](https://mlflow.org/)
* 💊 [LightGBM](https://lightgbm.readthedocs.io/)
* 🌐 [Streamlit](https://streamlit.io/)

---

## 🎯 Final Highlights

* [x] Modular and reproducible ML pipeline
* [x] Safe, incremental feature updates
* [x] Model versioning and performance monitoring
* [x] Automated CI/CD from data to dashboard
* [x] Public forecast and monitoring apps

---

## 🚀 Streamlit Apps

| 🔗 Link                                                   | Purpose                 |
| --------------------------------------------------------- | ----------------------- |
| 📈 [Prediction App](https://citibike-predictions-repo-6zyzt3rw6uiqk6eacxxyem.streamlit.app/))  | Live hourly predictions |
| 🧭 [Monitoring App](https://citibike-predictions-repo-vf7vmncg8qzw3meau7m3uk.streamlit.app/)   | MAE & model drift       |



---

## 💼 Author

Created with ❤️ by RedcondorAI
