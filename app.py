import os
import streamlit as st
import pandas as pd
import plotly.express as px
import hopsworks

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="Citi Bike Forecast Dashboard", layout="wide")

# ---------------- HOPSWORKS LOGIN ----------------
project = hopsworks.login(
    project=os.environ["HOPSWORKS_PROJECT_NAME"],
    api_key_value=os.environ["HOPSWORKS_API_KEY"],
    host="https://c.app.hopsworks.ai"
)
fs = project.get_feature_store()

# ---------------- LOAD FEATURE GROUPS ----------------
@st.cache_data(ttl=3600)
def load_feature_group(name: str, version: int):
    fg = fs.get_feature_group(name=name, version=version)
    return fg.read()

# Load all 6 datasets
df_lag28 = load_feature_group("citibike_predictions_lag28", version=1)
df_topk = load_feature_group("citibike_predictions_topk", version=1)
df_pca = load_feature_group("citibike_predictions_pca", version=1)

fut_lag28 = load_feature_group("citibike_forecast_lag28", version=1)
fut_topk = load_feature_group("citibike_forecast_topk", version=1)
fut_pca = load_feature_group("citibike_forecast_pca", version=1)

# ---------------- USER SELECTION ----------------
st.title("🚲 Citi Bike Forecast Dashboard")

model_option = st.selectbox("Choose a model", ["Lag-28", "Top-K", "PCA"])
station_option = st.selectbox("Choose a station", ["JC115", "HB102", "HB103"])

if model_option == "Lag-28":
    df_pred = df_lag28
    df_fut = fut_lag28
elif model_option == "Top-K":
    df_pred = df_topk
    df_fut = fut_topk
else:
    df_pred = df_pca
    df_fut = fut_pca

# Filter
df_pred = df_pred[df_pred["station_id"] == station_option]
df_fut = df_fut[df_fut["station_id"] == station_option]

# ---------------- PREDICTIONS ----------------
st.subheader(f"📊 Historical Prediction — {model_option} | {station_option}")
fig_pred = px.line(
    df_pred,
    x="hour",
    y=["actual_rides", "predicted_rides"],
    labels={"value": "rides", "hour": "Timestamp"},
    title="Actual vs Predicted"
)
st.plotly_chart(fig_pred, use_container_width=True)

# ---------------- FORECAST ----------------
st.subheader(f"🔮 Future Forecast — {model_option} | {station_option}")
fig_fut = px.line(
    df_fut,
    x="hour",
    y="predicted_rides",
    labels={"predicted_rides": "rides", "hour": "Future Timestamp"},
    title="Forecasted Rides (next 168 hours)"
)
st.plotly_chart(fig_fut, use_container_width=True)
