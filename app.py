import os
import streamlit as st
import pandas as pd
import plotly.express as px
import hopsworks

# ---------- CONFIG ----------
st.set_page_config(page_title="Citi Bike Forecast Dashboard", layout="wide")

# ---------- CUSTOM CSS ----------
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

# ---------- BACKGROUND IMAGE ----------
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
        .stMetric .st-emotion-cache-10trblm {{
            color: white;
        }}
        h1, h2, h3, p, span, div, label {{
            color: white !important;
        }}
        .css-18e3th9 {{
            padding-top: 2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Use a placeholder image URL (you can upload an image to your app and reference it)
set_background("https://images.unsplash.com/photo-1631217055910-b933d9ba3a2f?q=80&w=1974&auto=format&fit=crop")

# ---------- STATION + MODEL SETUP ----------
STATION_NAME_MAP = {
    "JC115": "Newport Parkway",
    "HB102": "Hoboken Terminal",
    "HB103": "14th Street Hoboken"
}

MODELS = {
    "Lag-28": {"pred": "citibike_predictions_lag28", "forecast": "citibike_forecast_lag28", "mae": "citibike_model_metrics_lag28"},
    "Top-K":  {"pred": "citibike_predictions_topk",  "forecast": "citibike_forecast_topk",  "mae": "citibike_model_metrics_topk"},
    "PCA":    {"pred": "citibike_predictions_pca",   "forecast": "citibike_forecast_pca",   "mae": "citibike_model_metrics_pca"}
}

# ---------- HOPSWORKS LOGIN ----------
@st.cache_resource
def get_hopsworks_connection():
    project = hopsworks.login(
        project=os.environ["HOPSWORKS_PROJECT_NAME"],
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        host="c.app.hopsworks.ai"
    )
    return project

# ---------- FEATURE GROUP LOADER ----------
@st.cache_data(ttl=3600)
def load_fg(name: str, version: int = 1):
    project = get_hopsworks_connection()
    fs = project.get_feature_store()
    return fs.get_feature_group(name=name, version=version).read()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("🚲 Citi Bike Forecast")
    
    st.markdown("### Settings")
    station_name = st.selectbox("Select Station", list(STATION_NAME_MAP.values()))
    station = [k for k, v in STATION_NAME_MAP.items() if v == station_name][0]
    
    model_option = st.selectbox("Select Model", list(MODELS.keys()))
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This dashboard visualizes Citi Bike hourly ridership.
    - Historical comparisons
    - Future forecasts
    - Model performance metrics
    """)

# ---------- TABS ----------
st.markdown("# 🚲 Citi Bike Analytics")
st.markdown(f"## {station_name} Station")

st.markdown("<div class='tab-style'>", unsafe_allow_html=True)
selected_tab = st.radio(
    "Navigation",
    ["Historical Predictions", "Future Forecast", "Model Summary"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown("</div>", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
try:
    df_pred = load_fg(MODELS[model_option]["pred"])
    df_pred = df_pred[df_pred["station_id"] == station]
    
    df_fore = load_fg(MODELS[model_option]["forecast"])
    df_fore = df_fore[df_fore["station_id"] == station]
    
    df_mae = load_fg(MODELS[model_option]["mae"])
    station_mae = df_mae[df_mae["station_id"] == station]["mae"].values[0]
except Exception as e:
    st.error(f"Error loading data from Hopsworks: {e}")
    st.stop()

# ---------- VIEWS ----------
if selected_tab == "Historical Predictions":
    st.subheader(f"📊 Historical: {model_option} Model")
    
    # Ensure hour is datetime and handle missing/non-numeric values
    if not pd.api.types.is_datetime64_dtype(df_pred["hour"]):
        df_pred["hour"] = pd.to_datetime(df_pred["hour"], errors='coerce')
    
    # Drop rows with NaT values in hour
    df_pred = df_pred.dropna(subset=["hour"])
    
    # Extract date only for daily grouping
    df_pred["date"] = df_pred["hour"].dt.date
    
    # Group by date (more resilient than resample)
    df_daily = df_pred.groupby("date").agg({
        "predicted_rides": "mean",
        "actual_rides": "mean"
    }).reset_index()
    
    # Convert date back to datetime for plotting
    df_daily["hour"] = pd.to_datetime(df_daily["date"])
    
    # Prepare for plotting
    df_plot = df_daily.copy()
    df_plot["Predicted Rides"] = df_plot["predicted_rides"]
    df_plot["Actual Rides"] = df_plot["actual_rides"]
    
    # Melt for Plotly
    plot_df = df_plot[["hour", "Predicted Rides", "Actual Rides"]].melt(
        id_vars="hour", var_name="Type", value_name="Rides"
    )
    
    # Create plot
    fig = px.line(
        plot_df,
        x="hour",
        y="Rides",
        color="Type",
        title=f"{model_option} — Actual vs Predicted for {station_name} (Daily Average)",
        template="plotly_dark"
    )
    
    # Customize plot
    custom_colors = {
        "Actual Rides": "#EF3E96",
        "Predicted Rides": "#4D96FF"
    }
    
    for trace in fig.data:
        trace.line.width = 2.5
        trace.opacity = 0.85
        trace.line.color = custom_colors.get(trace.name, trace.line.color)
    
    fig.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(size=18, color='white'),
        legend_title_text='',
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            tickformat="%b %d",
            tickangle=0,
            dtick="M1",  # Show one tick per month
            ticklabelmode="period"
        ),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # MAE metric
    st.metric("Mean Absolute Error (MAE)", f"{station_mae:.2f} rides")
    
    # Download option
    st.download_button(
        "📥 Download Results", 
        df_pred.to_csv(index=False), 
        file_name=f"{model_option}_{station}.csv"
    )

elif selected_tab == "Future Forecast":
    st.subheader(f"🔮 Forecast: {model_option} Model")
    
    # Make sure the date is in datetime format
    if not pd.api.types.is_datetime64_dtype(df_fore["hour"]):
        df_fore["hour"] = pd.to_datetime(df_fore["hour"], errors='coerce')
    
    # Drop rows with invalid datetime
    df_fore = df_fore.dropna(subset=["hour"])

    # Create 6-hour bins for aggregation
    df_fore["hour_bin"] = df_fore["hour"].dt.floor("6H")
    
    # Group by 6-hour intervals 
    df_fore_agg = df_fore.groupby("hour_bin").agg({
        "predicted_rides": "mean",
        "station_id": "first"  # Keep the station ID
    }).reset_index()
    
    # Rename columns for clarity
    df_fore_agg = df_fore_agg.rename(columns={"hour_bin": "hour"})
    
    fig = px.line(
        df_fore_agg,
        x="hour",
        y="predicted_rides",
        labels={"hour": "Time", "predicted_rides": "Predicted Rides"},
        title=f"{model_option} — 7-Day Forecast for {station_name} (6-Hour Intervals)",
        template="plotly_dark"
    )
    
    fig.update_traces(
        line=dict(color="#4D96FF", width=3),
        opacity=0.85
    )
    
    # Add markers to make data points clearer
    fig.update_traces(mode='lines+markers', marker=dict(size=6))
    
    fig.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(size=18, color='white'),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            tickformat="%b %d",
            tickangle=0,
            dtick="D1"  # One tick per day
        ),
        yaxis=dict(
            title="Predicted Rides"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Download option
    st.download_button(
        "📥 Download Forecast", 
        df_fore.to_csv(index=False), 
        file_name=f"{model_option}_forecast_{station}.csv"
    )

elif selected_tab == "Model Summary":
    st.subheader(f"📈 Model Performance Summary")
    
    # Get all model metrics for comparison
    all_mae_data = []
    for model_name, model_info in MODELS.items():
        # Add try/except blocks to handle potential errors
        try:
            mae_df = load_fg(model_info["mae"])
            station_data = mae_df[mae_df["station_id"] == station].copy()
            
            # Skip if no data for this station
            if len(station_data) == 0:
                continue
                
            station_data["model"] = model_name
            all_mae_data.append(station_data)
        except Exception as e:
            st.warning(f"Could not load metrics for {model_name}: {e}")
    
    if all_mae_data:
        # Combine all model data
        comparison_df = pd.concat(all_mae_data)
        comparison_df = comparison_df.sort_values("mae")
        
        # # Display as a table
        # st.dataframe(
        #     comparison_df[["model", "station_id", "mae"]].rename(
        #         columns={"model": "Model", "station_id": "Station", "mae": "MAE"}
        #     ),
        #     use_container_width=True
        # )
        
        # Prepare ranked display table
        display_df = comparison_df[["model", "station_id", "mae"]].rename(
            columns={"model": "Model", "station_id": "Station", "mae": "MAE"}
        )
        display_df["Rank"] = range(1, len(display_df) + 1)
        st.dataframe(display_df.set_index("Rank"), use_container_width=True)

        
        
        # Create a bar chart comparison
        fig = px.bar(
            comparison_df,
            x="model",
            y="mae",
            title=f"MAE Comparison for {station_name}",
            template="plotly_dark",
            color="model",
            labels={"model": "Model", "mae": "Mean Absolute Error"}
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font=dict(size=18, color='white'),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        best_model = comparison_df.iloc[0]["model"]
        best_mae = comparison_df.iloc[0]["mae"]
        st.success(f"✅ Best model for {station_name}: **{best_model}** with MAE of **{best_mae:.2f}**")
    else:
        st.error("No model metrics data available")