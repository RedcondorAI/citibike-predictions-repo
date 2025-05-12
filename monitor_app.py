import os
import streamlit as st
import pandas as pd
import plotly.express as px
import hopsworks

# ---------- CONFIG ----------
st.set_page_config(page_title="Model Monitoring - Citi Bike", layout="wide")

# ---------- CUSTOM CSS ----------
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

# ---------- SETTINGS ----------
COLOR_BLUE = "#00AEEF"
COLOR_PINK = "#EF3E96"
COLOR_NAVY = "#0033A0"
COLOR_TEAL = "#00B2A9"

STATION_NAME_MAP = {
    "JC115": "Newport Parkway",
    "HB102": "Hoboken Terminal",
    "HB103": "14th Street Hoboken"
}

# Match the exact feature group names from your Hopsworks setup
MODELS = {
    "Lag-28 Model": {
        "metrics": "citibike_model_metrics_lag28",
        "predictions": "citibike_predictions_lag28",
        "forecast": "citibike_forecast_lag28",
        "strategy": "All Lags",
        "color": COLOR_BLUE
    },
    "Top-K Model": {
        "metrics": "citibike_model_metrics_topk",
        "predictions": "citibike_predictions_topk",
        "forecast": "citibike_forecast_topk",
        "strategy": "Top 10 Features",
        "color": COLOR_PINK
    },
    "PCA Model": {
        "metrics": "citibike_model_metrics_pca",
        "predictions": "citibike_predictions_pca",
        "forecast": "citibike_forecast_pca",
        "strategy": "PCA Reduction",
        "color": COLOR_TEAL
    }
    # Removed Naive Model since it doesn't have prediction data in Hopsworks
    # You can uncomment if you add this feature group later
    # "Naive Model": {
    #     "metrics": "citibike_model_metrics_baseline",
    #     "predictions": "citibike_predictions_baseline", 
    #     "forecast": "citibike_forecast_baseline",
    #     "strategy": "Naive",
    #     "color": COLOR_NAVY
    # }
}

# ---------- HOPSWORKS CONNECTION ----------
@st.cache_resource
def get_hopsworks_connection():
    try:
        project = hopsworks.login(
            project=os.environ["HOPSWORKS_PROJECT_NAME"],
            api_key_value=os.environ["HOPSWORKS_API_KEY"],
            host="c.app.hopsworks.ai"
        )
        return project
    except Exception as e:
        st.error(f"Failed to connect to Hopsworks: {str(e)}")
        st.error("Please check your environment variables: HOPSWORKS_PROJECT_NAME and HOPSWORKS_API_KEY")
        raise e

# ---------- FEATURE GROUP LOADER ----------
@st.cache_data(ttl=3600)
def load_fg(name: str, version: int = 1):
    try:
        project = get_hopsworks_connection()
        fs = project.get_feature_store()
        return fs.get_feature_group(name=name, version=version).read()
    except Exception as e:
        st.warning(f"Could not load feature group {name}: {e}")
        return pd.DataFrame()

# ---------- Load MAE Metrics ----------
def load_all_metrics():
    dfs = {}
    for model_name, model_info in MODELS.items():
        try:
            df = load_fg(model_info["metrics"])
            if not df.empty:
                df["Model"] = model_name
                dfs[model_name] = df
        except Exception as e:
            st.warning(f"Error loading metrics for {model_name}: {e}")
    return dfs

# ---------- Load Time Series Data ----------
def load_timeseries_predictions(is_forecast, station):
    dfs = []
    for model_name, model_info in MODELS.items():
        try:
            # Skip naive/baseline model if feature group doesn't exist
            if model_name == "Naive Model" and not is_forecast:
                continue
                
            feature_group = model_info["forecast"] if is_forecast else model_info["predictions"]
            df = load_fg(feature_group)
            
            if not df.empty:
                # Filter by station
                df = df[df["station_id"] == station]
                
                # Skip if no data for this station
                if df.empty:
                    continue
                
                # Make sure 'hour' is datetime
                if not pd.api.types.is_datetime64_dtype(df["hour"]):
                    df["hour"] = pd.to_datetime(df["hour"], errors='coerce')
                
                # Drop rows with NaT values in hour
                df = df.dropna(subset=["hour"])
                
                # Convert to pandas datetime without timezone for consistent comparison
                df["hour"] = df["hour"].dt.tz_localize(None)
                
                # Filter historical data for the past 60 days
                if not is_forecast:
                    current_time = pd.Timestamp.now()
                    sixty_days_ago = current_time - pd.Timedelta(days=60)
                    df = df[df['hour'] >= sixty_days_ago]
                
                # Rename column and add model information
                df = df.rename(columns={"predicted_rides": "Rides"})
                df["Model"] = model_name
                
                # Select relevant columns
                dfs.append(df[["hour", "Rides", "Model"]])
        except Exception as e:
            st.warning(f"Error loading {model_name} prediction data: {e}")
    
    return pd.concat(dfs) if dfs else pd.DataFrame()

# ---------- Main App ----------
def main():
    # Add an expander for debugging connection information
    # with st.expander("Connection Information", expanded=False):
    #     st.markdown("""
    #     ### Hopsworks Connection
    #     This app connects to Hopsworks for data retrieval. Make sure you have set the following environment variables:
    #     - `HOPSWORKS_PROJECT_NAME`
    #     - `HOPSWORKS_API_KEY`
        
    #     If you're experiencing connection issues, check your API key and project name.
    #     """)
        
    #     if st.button("Test Hopsworks Connection"):
    #         try:
    #             project = get_hopsworks_connection()
    #             st.success(f"✅ Successfully connected to Hopsworks project: {project.name}")
    #         except Exception as e:
    #             st.error(f"❌ Connection failed: {str(e)}")
    
    metrics_dict = load_all_metrics()
    if not metrics_dict:
        st.error("No metrics found in Hopsworks feature store.")
        return

    df_all = pd.concat(metrics_dict.values(), ignore_index=True)
    df_all["Station Name"] = df_all["station_id"].map(STATION_NAME_MAP)
    
    # Add strategy information from our models dictionary
    df_all["Strategy"] = df_all.apply(lambda row: MODELS.get(row["Model"], {}).get("strategy", "Unknown"), axis=1)

    unique_stations = df_all["Station Name"].dropna().unique().tolist()
    selected_stations = st.multiselect("Select Station(s) to Compare", unique_stations, default=unique_stations)

    df_filtered = df_all[df_all["Station Name"].isin(selected_stations)]

    # ---------- MAE Grouped Bar Chart ----------
    if not df_filtered.empty:
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
                model_name: model_info["color"] for model_name, model_info in MODELS.items()
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

        # ---------- MAE Table ----------
        df_clean = df_filtered.rename(columns={"mae": "MAE"})[
            ["Model", "Station Name", "Strategy", "MAE"]
        ]
        
        df_clean = df_clean.sort_values("staion_id","mae")
        df_clean["Rank"] = range(1, len(df_clean) + 1)
        st.dataframe(df_clean.set_index("Rank"), use_container_width=True)

        csv = df_clean.to_csv(index=True).encode("utf-8")
        st.download_button("Download MAE Summary as CSV", csv, "mae_summary_all_models.csv")

        # ---------- Time Series Predictions / Forecast ----------
        if len(selected_stations) == 1:
            station_code = [k for k, v in STATION_NAME_MAP.items() if v == selected_stations[0]][0]

            st.subheader("Historical Predictions (Last 60 Days)")
            past_df = load_timeseries_predictions(is_forecast=False, station=station_code)
            
            if not past_df.empty:
                # Resample to daily averages to reduce noise
                past_df['date'] = past_df['hour'].dt.date
                daily_past_df = past_df.groupby(['date', 'Model']).agg({'Rides': 'mean'}).reset_index()
                daily_past_df['hour'] = pd.to_datetime(daily_past_df['date'])
                
                fig1 = px.line(
                    daily_past_df, 
                    x="hour", 
                    y="Rides", 
                    color="Model",
                    title="Predictions by Model - Last 60 Days (Daily Average)",
                    color_discrete_map={
                        model_name: model_info["color"] for model_name, model_info in MODELS.items()
                    }
                )
                fig1.update_layout(
                    font=dict(color='white'),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                        title="Date",
                        tickformat="%b %d",
                        tickangle=0,
                    ),
                    yaxis=dict(title="Average Rides")
                )
                # Make lines thicker for better visibility
                fig1.update_traces(line=dict(width=2.5))
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("No historical prediction data available for the selected station.")

            st.subheader("Forecasts (Next 7 Days)")
            future_df = load_timeseries_predictions(is_forecast=True, station=station_code)
            
            if not future_df.empty:
                # Resample to 6-hour intervals to reduce noise and make trends clearer
                future_df['hour_bin'] = future_df['hour'].dt.floor('6H')
                future_resampled = future_df.groupby(['hour_bin', 'Model']).agg({'Rides': 'mean'}).reset_index()
                
                fig2 = px.line(
                    future_resampled, 
                    x="hour_bin", 
                    y="Rides", 
                    color="Model",
                    title="Forecast by Model - Next 7 Days (6-Hour Intervals)",
                    color_discrete_map={
                        model_name: model_info["color"] for model_name, model_info in MODELS.items()
                    }
                )
                fig2.update_layout(
                    font=dict(color='white'),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                        title="Date",
                        tickformat="%b %d",
                        dtick="D1"  # One tick per day
                    ),
                    yaxis=dict(title="Predicted Rides")
                )
                # Make lines thicker and add markers for better visualization
                fig2.update_traces(
                    line=dict(width=3),
                    mode='lines+markers',
                    marker=dict(size=6)
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No forecast data available for the selected station.")
    else:
        st.warning("No data available for the selected stations.")

if __name__ == "__main__":
    main()