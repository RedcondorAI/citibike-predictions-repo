# 🚲 Citi Bike Trip Prediction System

![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)
![License](https://img.shields.io/github/license/yourusername/citibike-predictions)
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

## 📁 Directory Structure

```
citibike-predictions/
├── data/                      # Raw, processed, and prediction data
├── trained_models/           # Pickled LightGBM/Naive/PCA models
├── src/
│   ├── data/                 # Data fetching & preprocessing
│   ├── features/             # Feature engineering scripts
│   ├── models/               # Baseline, Lag28, TopK, PCA
│   ├── forecast/             # Future forecasts (7-day)
│   ├── upload/               # Hopsworks upload scripts
│   └── utils/                # MLflow logger
├── app/                      # Streamlit dashboard apps
├── .github/workflows/        # GitHub Actions (CI automation)
├── .env                      # API keys (not tracked)
├── requirements.txt
└── README.md
```

---

## 📘 Naming Conventions

| Component        | Format Example                           |
| ---------------- | ---------------------------------------- |
| Model files      | `lgbm_lag28_model_JC115.pkl`             |
| Prediction files | `future_lgbm_topk_HB102.csv`             |
| Hopsworks FG     | `citibike_features_dataset`              |
| Prediction FG    | `citibike_predictions_dataset_jc115`     |
| MLflow Names     | `citibike-lgbm-pca`, `citibike-baseline` |

---

## 🔧 Local Setup

### Step 1: Clone and Install

```bash
git clone https://github.com/yourusername/citibike-predictions.git
cd citibike-predictions
conda create -n citibike python=3.10
conda activate citibike
pip install -r requirements.txt
```

### Step 2: Configure `.env`

```env
HOPSWORKS_PROJECT_NAME=your_project_name
HOPSWORKS_API_KEY=your_api_key
MLFLOW_TRACKING_URI=https://dagshub.com/youruser/yourrepo.mlflow
```

---

## 📥 Data Pipeline

### ✅ 1. Fetch & Clean Data

```bash
python src/data/fetch_data.py
python src/data/preprocess_data.py
```

### 🧠 2. Engineer Features

```bash
python src/features/engineering_features.py
```

---

## 🔀 Model Training

Run any of the following:

| Model                | Script                              |
| -------------------- | ----------------------------------- |
| Naive (baseline)     | `src/models/baseline_model.py`      |
| Full Lag-28 LightGBM | `src/models/lightgbm_model.py`      |
| Top-K Feature Model  | `src/models/lightgbm_topk_model.py` |
| PCA-based Model      | `src/models/lightgbm_pca_model.py`  |

---

## 🔮 Inference & Forecasting

Predict future rides (7-day horizon):

```bash
python src/forecast/forecast_future_lag28.py
python src/forecast/forecast_future_topk.py
python src/forecast/forecast_future_pca.py
```

---

## ☁️ Upload to Hopsworks

Upload features, models, and predictions:

```bash
python src/upload/upload_to_hopsworks_best.py
python src/upload/upload_to_hopsworks_other.py
python src/upload/upload_to_hopsworks_pca.py
```

---

## ⚙️ GitHub Actions Automation

| Workflow        | Purpose                       |
| --------------- | ----------------------------- |
| `train.yml`     | Retrain + upload all models   |
| `inference.yml` | Forecast + upload predictions |

To run manually:

```bash
gh workflow run train.yml
```

---

## 📊 Streamlit Dashboards

| 🔗 Link                                                      | Purpose               |
| ------------------------------------------------------------ | --------------------- |
| 🚀 [Forecast App](https://your-forecast-app.streamlit.app/)  | Live ride predictions |
| 📉 [Monitoring App](https://your-monitor-app.streamlit.app/) | Model performance     |

To launch locally:

```bash
streamlit run app/app.py
```

---

## 📈 MLflow Logging

Each model run logs:

* MAE, strategy, model type
* Feature count / PCA explained variance
* Signatures + input examples

Visit: [MLflow Tracking UI](https://dagshub.com/youruser/yourrepo.mlflow)

---

## ✅ Final Checklist

* [x] Historical data (2023–2025) used
* [x] Models logged via MLflow
* [x] Hopsworks upload working
* [x] Streamlit dashboards functional
* [x] GitHub Actions integrated

---

## 🧪 Quick Test

```bash
python src/models/baseline_model.py
python src/forecast/forecast_future_lag28.py
```

---

## 📌 Data & Credits

* 🚲 [Citi Bike NYC Trip Data](https://citibikenyc.com/system-data)
* 🌐 [Hopsworks Feature Store](https://www.hopsworks.ai/)
* 🔬 [MLflow Tracking](https://mlflow.org/)
* 🌟 Icons by [SimpleIcons](https://simpleicons.org/)

---

## 👤 Author & Contact

Created with ❤️ by \[Your Name].
Open an issue or contact via GitHub for questions.

---
