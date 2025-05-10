import streamlit as st
import pandas as pd
import os
import plotly.express as px

# --- CONFIG ---
st.set_page_config(page_title="Citi Bike Dashboard", layout="wide")

st.markdown("""
    <style>
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


# --- BACKGROUND IMAGE ---
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)), url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Use your chosen image
set_background("https://images.unsplash.com/photo-1631217055910-b933d9ba3a2f?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")

# --- STATION + MODEL SETUP ---
STATION_NAME_MAP = {
    "JC115": "Newport Parkway",
    "HB102": "Hoboken Terminal",
    "HB103": "14th Street Hoboken"
}

MODELS = {
    "Baseline": "predictions_naive_{}.csv",
    "Lag-28": "predictions_lgbm_lag28_{}.csv",   
    "TopK": "predictions_lgbm_topk_{}.csv",      
    "PCA": "predictions_lgbm_pca_{}.csv"         
}

FORECAST_FILES = {
    "Lag-28": "future_lgbm_lag28_{}.csv",
    "TopK": "future_lgbm_topk_{}.csv",
    "PCA": "future_lgbm_pca_{}.csv"
}
METRICS_PATH = "../data/metrics"

# --- CACHE ---
@st.cache_data
def load_csv(path):
    return pd.read_csv(path, parse_dates=["hour"])

# --- SIDEBAR ---
station_name = st.sidebar.selectbox("Select Station", list(STATION_NAME_MAP.values()))
station = [k for k, v in STATION_NAME_MAP.items() if v == station_name][0]
model = st.sidebar.radio("Select Model", list(MODELS.keys()))
model_file = os.path.normpath(os.path.join(METRICS_PATH, MODELS[model].format(station)))

# --- TABS ---

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='tab-style'>", unsafe_allow_html=True)
selected_tab = st.radio(
    "Navigation",
    ["Historical Predictions", "Forecast", "Model Summary"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown("</div>", unsafe_allow_html=True)

#tab1, tab2, tab3 = st.tabs(["📊 Historical Predictions", "🔮 Forecast", "📈 Model Summary"])

# --- TAB 1: HISTORICAL PLOT ---
if selected_tab == "Historical Predictions":
    if os.path.exists(model_file):
        df = pd.read_csv(model_file, parse_dates=["hour"])

        # Rename + smooth
        df["Predicted Rides"] = df["predicted_rides"].rolling(6).mean()
        if "actual_rides" not in df.columns and "rides" in df.columns:
            df["Actual Rides"] = df["rides"].rolling(6).mean()
        elif "actual_rides" in df.columns:
            df["Actual Rides"] = df["actual_rides"].rolling(6).mean()

        # Melt for Plotly
        cols_to_plot = ["Predicted Rides"]
        if "Actual Rides" in df.columns:
            cols_to_plot.append("Actual Rides")
        plot_df = df[["hour"] + cols_to_plot].melt(id_vars="hour", var_name="Type", value_name="Rides")

        # Plotly Chart
        fig = px.line(
            plot_df,
            x="hour",
            y="Rides",
            color="Type",
            title=f"{model} — Actual vs Predicted for {station_name} (Smoothed)",
            template="plotly_dark"
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font=dict(size=18, color='white'),
            legend_title_text='',
            margin=dict(l=20, r=20, t=40, b=20)
        )

        # Custom colors
        custom_colors = {
            "Actual Rides": "#EF3E96",
            "Predicted Rides": "#4D96FF"
        }
        for trace in fig.data:
            trace.line.width = 2
            trace.opacity = 0.85
            trace.line.color = custom_colors.get(trace.name, trace.line.color)

        st.plotly_chart(fig, use_container_width=True)

        # MAE
        if "actual_rides" in df.columns:
            mae = abs(df["actual_rides"] - df["predicted_rides"]).mean()
            st.metric("Mean Absolute Error (MAE)", f"{mae:.2f} rides")

        st.download_button("📥 Download Results", df.to_csv(index=False), file_name=f"{model}_{station}.csv")
    else:
        st.error(f"Prediction file not found: {model_file}")

# --- TAB 2: FORECAST ---
elif selected_tab == "Forecast":
    if model in FORECAST_FILES:
        forecast_path = os.path.normpath(os.path.join(METRICS_PATH, FORECAST_FILES[model].format(station)))
        if os.path.exists(forecast_path):
            fdf = pd.read_csv(forecast_path, parse_dates=["hour"])

            fig = px.line(
                fdf,
                x="hour",
                y="predicted_rides",
                labels={"hour": "Time", "predicted_rides": "Predicted Rides"},
                title=f"{model} — 7-Day Forecast for {station_name}",
                template="plotly_dark"
            )

            fig.update_traces(
                line=dict(color="#4D96FF", width=2),
                opacity=0.85
            )

            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No forecast data available for {model}.")
    else:
        st.info("Forecasting not available for the Baseline model.")


# --- TAB 3: MAE TABLE ---
elif selected_tab == "Model Summary":
    st.markdown(f"**Model Model Summary for {station_name}**")

    mae_files = [
        "baseline_mae_summary.csv",
        "lgbm_lag28_mae_summary.csv",
        "lgbm_topk_mae_summary.csv",
        "lgbm_pca_mae_summary.csv"
    ]

    display_model_names = {
        "LGBMLag28": "LightGBM (Lag-28)",
        "LGBMPCA": "LightGBM (PCA)",
        "LGBMTopK": "LightGBM (Top-K)",
        "naive_lag_1": "Naive (Lag-1)"
    }

    display_strategies = {
        "LGBMLag28": "All Lags",
        "LGBMTopK": "Top 10 Features",
        "LGBMPCA": "PCA Reduction",
        "naive_lag_1": "Naive"
    }

    mae_frames = []
    for file in mae_files:
        path = os.path.normpath(os.path.join(METRICS_PATH, file))
        if os.path.exists(path):
            mdf = pd.read_csv(path)
            if "model" in mdf.columns:
                mdf["strategy"] = mdf["model"].map(display_strategies).fillna("")
                mdf["model"] = mdf["model"].map(display_model_names).fillna(mdf["model"])
            mae_frames.append(mdf)

    if mae_frames:
        mae_df = pd.concat(mae_frames)
        filtered = mae_df[mae_df["station_id"] == station].copy()
        filtered = filtered.drop(columns=["station_id"])
        filtered = filtered.rename(columns={
            "model": "Model",
            "strategy": "Strategy",
            "explained_variance": "Explained Variance (%)"
        })

        if "mae" in filtered.columns:
            filtered["MAE"] = filtered["mae"].map(lambda x: round(x, 2))
            filtered = filtered.drop(columns=["mae"])

        filtered = filtered.sort_values("MAE")
        filtered.reset_index(drop=True, inplace=True)
        filtered.index += 1

        st.dataframe(filtered, use_container_width=True)
    else:
        st.warning("No MAE data found.")
