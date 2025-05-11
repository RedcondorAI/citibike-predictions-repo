import os
import streamlit as st
import pandas as pd
import plotly.express as px
import hopsworks

# ---------- SETUP ----------
st.set_page_config(page_title="Citi Bike Forecast Dashboard", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
    }
    .tab-style .stRadio > div {
        display: flex;
        justify-content: center;
        gap: 1rem;
    }
    .tab-style label {
        background-color: #2F2F2F;
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    .tab-style label:hover {
        border-color: #EF3E96;
        color: #EF3E96;
    }
    .tab-style input:checked + div label {
        background-color: #EF3E96 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- HOPSWORKS LOGIN ----------
project = hopsworks.login(
    project=os.environ["HOPSWORKS_PROJECT_NAME"],
    api_key_value=os.environ["HOPSWORKS_API_KEY"],
    host="c.app.hopsworks.ai"
)
fs = project.get_feature_store()

# ---------- FEATURE GROUP LOADER ----------
@st.cache_data(ttl=3600)
def load_fg(name: str, version: int = 1):
    return fs.get_feature_group(name=name, version=version).read()

models = {
    "Lag-28": {"pred": "citibike_predictions_lag28", "forecast": "citibike_forecast_lag28", "mae": "citibike_model_metrics_lag28"},
    "Top-K":  {"pred": "citibike_predictions_topk",  "forecast": "citibike_forecast_topk",  "mae": "citibike_model_metrics_topk"},
    "PCA":    {"pred": "citibike_predictions_pca",   "forecast": "citibike_forecast_pca",   "mae": "citibike_model_metrics_pca"}
}

# ---------- UI ----------
st.title("🚲 Citi Bike Forecast Dashboard")

with st.container():
    st.markdown("##")
    tab = st.radio("Choose View", ["Prediction vs Actual", "Future Forecast", "MAE Summary"], horizontal=True, label_visibility="collapsed")
    model_option = st.selectbox("Choose Model", list(models.keys()))
    station_option = st.selectbox("Choose Station", ["JC115", "HB102", "HB103"])

# ---------- LOAD + FILTER DATA ----------
df_pred = load_fg(models[model_option]["pred"])
df_fore = load_fg(models[model_option]["forecast"])
df_mae  = load_fg(models[model_option]["mae"])

df_pred = df_pred[df_pred["station_id"] == station_option]
df_fore = df_fore[df_fore["station_id"] == station_option]
mae_val = df_mae[df_mae["station_id"] == station_option]["mae"].values[0]

# ---------- VIEWS ----------
if tab == "Prediction vs Actual":
    st.subheader(f"📊 Historical: {model_option} — {station_option} (MAE: {mae_val:.2f})")

    # Group to avoid clutter
    df_plot = df_pred.groupby("hour")[["actual_rides", "predicted_rides"]].mean().reset_index()

    fig = px.line(
        df_plot,
        x="hour",
        y="actual_rides",
        title="Actual vs Predicted",
        labels={"actual_rides": "Rides", "hour": "Time"}
    )
    fig.add_scatter(
        x=df_plot["hour"],
        y=df_plot["predicted_rides"],
        mode='lines',
        name="Predicted"
    )
    fig.update_layout(
        xaxis=dict(tickformat="%b %Y", tickangle=0, dtick="M1"),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

elif tab == "Future Forecast":
    st.subheader(f"🔮 Forecast: {model_option} — {station_option}")
    fig = px.line(df_fore, x="hour", y="predicted_rides", labels={"predicted_rides": "Rides", "hour": "Time"}, title="Future Forecast (Next 168 Hours)")
    fig.update_layout(xaxis=dict(tickformat="%b %d", tickangle=45))
    st.plotly_chart(fig, use_container_width=True)

elif tab == "MAE Summary":
    st.subheader(f"📉 MAE Summary — {model_option}")
    st.metric(label=f"Station {station_option} MAE", value=f"{mae_val:.2f}")
