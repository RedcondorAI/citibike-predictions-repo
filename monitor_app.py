import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- CONFIG ---
st.set_page_config(page_title="Model Monitoring - Citi Bike", layout="wide")

# --- BACKGROUND IMAGE + THEME FIX ---
st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), url("https://images.pexels.com/photos/15647073/pexels-photo-15647073/free-photo-of-a-bicycle-for-rental-parked-in-the-bicycle-stand-near-a-park.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    h1, h2, h3, label, .css-1v3fvcr {
        color: white !important;
    }
    .stMultiSelect>div>div {
        background-color: rgba(0,174,239,0.2) !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Citi Bike Model Monitoring Dashboard")

# --- SETTINGS ---
METRICS_DIR = "../data/metrics"
COLOR_BLUE = "#00AEEF"
COLOR_PINK = "#EF3E96"
COLOR_NAVY = "#0033A0"
COLOR_TEAL = "#00B2A9"

STATION_NAME_MAP = {
    "JC115": "Newport Parkway",
    "HB102": "Hoboken Terminal",
    "HB103": "14th Street Hoboken"
}

DISPLAY_STRATEGIES = {
    "LGBMLag28": "All Lags",
    "LGBMTopK": "Top 10 Features",
    "LGBMPCA": "PCA Reduction",
    "naive_lag_1": "Naive"
}

# --- Load MAE Summary Files ---
def load_metrics():
    dfs = {}
    files = [
        ("Lag-28 Model", "lgbm_lag28_mae_summary.csv"),
        ("Top-K Model", "lgbm_topk_mae_summary.csv"),
        ("PCA Model", "lgbm_pca_mae_summary.csv"),
        ("Naive Model", "baseline_mae_summary.csv"),
    ]
    for name, fname in files:
        path = os.path.join(METRICS_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["Model"] = name
            dfs[name] = df
    return dfs

# --- Load Time Series Data ---
def load_timeseries_predictions(prefix, station):
    model_files = {
        "Lag-28 Model": f"{prefix}lgbm_lag28_{station}.csv",
        "Top-K Model": f"{prefix}lgbm_topk_{station}.csv",
        "PCA Model": f"{prefix}lgbm_pca_{station}.csv",
        "Naive Model": f"{prefix}naive_{station}.csv" if prefix == "" else None
    }
    dfs = []
    for model, file in model_files.items():
        if file is None:
            continue
        path = os.path.join(METRICS_DIR, file)
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["hour"])
            df = df.rename(columns={"predicted_rides": "Rides"})
            df = df[df['hour'] >= pd.Timestamp.now() - pd.Timedelta(days=60)]  # Filter past 7 days
            df["Model"] = model
            dfs.append(df[["hour", "Rides", "Model"]])
    return pd.concat(dfs) if dfs else pd.DataFrame()

# --- Main App ---
def main():
    metrics_dict = load_metrics()
    if not metrics_dict:
        st.error("No metrics found in ../data/metrics/")
        return

    df_all = pd.concat(metrics_dict.values(), ignore_index=True)
    df_all["Station Name"] = df_all["station_id"].map(STATION_NAME_MAP)
    df_all["Strategy"] = df_all["model"].map(DISPLAY_STRATEGIES)

    unique_stations = df_all["Station Name"].dropna().unique().tolist()
    selected_stations = st.multiselect("Select Station(s) to Compare", unique_stations, default=unique_stations)

    df_filtered = df_all[df_all["Station Name"].isin(selected_stations)]

    # --- MAE Grouped Bar Chart ---
    fig = px.bar(
        df_filtered,
        x="Station Name",
        y="mae",
        color="Model",
        text="mae",
        barmode="group",
        title="Mean Absolute Error (MAE) by Station and Model",
        labels={"mae": "MAE"},
        color_discrete_map={
            "Lag-28 Model": COLOR_BLUE,
            "Top-K Model": COLOR_PINK,
            "PCA Model": COLOR_TEAL,
            "Naive Model": COLOR_NAVY,
        }
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        title_font_size=22,
        legend_title_text="Model",
        bargap=0.25,
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- MAE Table ---
    df_clean = df_filtered.rename(columns={"mae": "MAE", "model": "Raw Model"})[
        ["Model", "Station Name", "Strategy", "MAE"]
    ]
    df_clean.index = df_clean.index + 1
    st.dataframe(df_clean, use_container_width=True)

    csv = df_clean.to_csv(index=True).encode("utf-8")
    st.download_button("Download MAE Summary as CSV", csv, "mae_summary_all_models.csv")

    # --- Time Series Predictions / Forecast ---
    if len(selected_stations) == 1:
        station_code = [k for k, v in STATION_NAME_MAP.items() if v == selected_stations[0]][0]

        st.subheader("Historical Predictions (Last 60 Days)")
        past_df = load_timeseries_predictions("predictions_", station_code)
        past_df = past_df[past_df['hour'] >= pd.Timestamp.now() - pd.Timedelta(days=60)]
        if not past_df.empty:
            fig1 = px.line(past_df, x="hour", y="Rides", color="Model",
                           title="Predictions by Model - Last 7 Days",
                           color_discrete_map={
                               "Lag-28 Model": COLOR_BLUE,
                               "Top-K Model": COLOR_PINK,
                               "PCA Model": COLOR_TEAL,
                               "Naive Model": COLOR_NAVY,
                           })
            fig1.update_layout(
                font=dict(color='white'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Forecasts (Next 7 Days)")
        future_df = load_timeseries_predictions("future_", station_code)
        if not future_df.empty:
            fig2 = px.line(future_df, x="hour", y="Rides", color="Model",
                           title="Forecast by Model - Next 7 Days",
                           color_discrete_map={
                               "Lag-28 Model": COLOR_BLUE,
                               "Top-K Model": COLOR_PINK,
                               "PCA Model": COLOR_TEAL,
                               "Naive Model": COLOR_NAVY,
                           })
            fig2.update_layout(
                font=dict(color='white'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()
