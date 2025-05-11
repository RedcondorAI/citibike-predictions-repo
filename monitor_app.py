import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import hopsworks
from datetime import datetime, timedelta

# --- CONFIG ---
st.set_page_config(
    page_title="Model Monitoring - Citi Bike", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- BACKGROUND IMAGE + THEME FIX ---
st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url("https://images.pexels.com/photos/15647073/pexels-photo-15647073/free-photo-of-a-bicycle-for-rental-parked-in-the-bicycle-stand-near-a-park.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    h1, h2, h3, h4, label, .css-1v3fvcr, .st-bo, p {
        color: white !important;
    }
    .stMultiSelect>div>div {
        background-color: rgba(0,174,239,0.2) !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(0,0,0,0.3);
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(239,62,150,0.7) !important;
        border-bottom: 2px solid white;
    }
    div[data-testid="stToolbar"] {
        display: none;
    }
    .metric-card {
        background-color: rgba(0,0,0,0.3);
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #EF3E96;
    }
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        background-color: rgba(0,0,0,0.2);
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SETTINGS ---
COLOR_BLUE = "#00AEEF"
COLOR_PINK = "#EF3E96"
COLOR_NAVY = "#0033A0"
COLOR_TEAL = "#00B2A9"

STATION_NAME_MAP = {
    "JC115": "Newport Parkway",
    "HB102": "Hoboken Terminal",
    "HB103": "14th Street Hoboken"
}

MODEL_CONFIGS = {
    "Lag-28 Model": {
        "metrics": "citibike_model_metrics_lag28",
        "predictions": "citibike_predictions_lag28",
        "forecast": "citibike_forecast_lag28",
        "drift": "citibike_drift_metrics_lag28",
        "color": COLOR_BLUE
    },
    "Top-K Model": {
        "metrics": "citibike_model_metrics_topk",
        "predictions": "citibike_predictions_topk",
        "forecast": "citibike_forecast_topk",
        "drift": "citibike_drift_metrics_topk",
        "color": COLOR_PINK
    },
    "PCA Model": {
        "metrics": "citibike_model_metrics_pca",
        "predictions": "citibike_predictions_pca",
        "forecast": "citibike_forecast_pca",
        "drift": "citibike_drift_metrics_pca",
        "color": COLOR_TEAL
    },
    "Naive Model": {
        "metrics": "citibike_model_metrics_naive",
        "predictions": "citibike_predictions_naive",
        "forecast": "citibike_forecast_naive",
        "drift": "citibike_drift_metrics_naive",
        "color": COLOR_NAVY
    }
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
        st.error(f"Failed to connect to Hopsworks: {e}")
        return None

# ---------- DATA LOADING FUNCTIONS ----------
@st.cache_data(ttl=3600)
def load_feature_group(name, version=1):
    """Load a feature group from Hopsworks"""
    try:
        project = get_hopsworks_connection()
        if project is None:
            return pd.DataFrame()
            
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=name, version=version)
        return fg.read()
    except Exception as e:
        st.warning(f"Could not load feature group '{name}': {e}")
        return pd.DataFrame()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("🚲 Citi Bike Monitoring")
    
    # Show status of Hopsworks connection
    project = get_hopsworks_connection()
    if project:
        st.success("✅ Connected to Hopsworks")
    else:
        st.error("❌ Not connected to Hopsworks")
        st.info("Set HOPSWORKS_PROJECT_NAME and HOPSWORKS_API_KEY in environment variables")
        
    st.markdown("### Configuration")
    
    # Time period selection for historical data
    time_periods = {
        "Last 7 days": 7,
        "Last 30 days": 30,
        "Last 60 days": 60
    }
    selected_period = st.selectbox("Time Period", list(time_periods.keys()), index=1)
    days_to_show = time_periods[selected_period]
    
    # Model selection for metrics and drift analysis
    selected_models = st.multiselect(
        "Select Models",
        list(MODEL_CONFIGS.keys()),
        default=list(MODEL_CONFIGS.keys())[:3]  # Default to first 3 models
    )
    
    st.markdown("### About")
    st.markdown("""
    This dashboard monitors model performance metrics and data drift for Citi Bike prediction models.
    
    Key metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - Feature Drift: Changes in distribution of key features
    """)
    
    # Add download button for sample data
    st.markdown("### Documentation")
    st.markdown("[Visit Documentation](https://github.com/yourusername/citibike-ml)")

# ---------- MAIN CONTENT ----------
st.title("Citi Bike Model Monitoring Dashboard")
st.caption("Real-time monitoring of model performance and data drift")

# Create tabs for different monitoring aspects
tabs = st.tabs([
    "🎯 Performance Metrics", 
    "📈 Predictions vs Actuals", 
    "🔮 Future Forecasts",
    "🔄 Data Drift"
])

# ---------- PERFORMANCE METRICS TAB ----------
with tabs[0]:
    st.header("Model Performance Metrics")
    
    # Load metrics for each model and combine
    metrics_dfs = []
    for model_name in selected_models:
        if model_name in MODEL_CONFIGS:
            metrics_fg = MODEL_CONFIGS[model_name]["metrics"]
            df = load_feature_group(metrics_fg)
            if not df.empty:
                df["model"] = model_name
                metrics_dfs.append(df)
    
    if not metrics_dfs:
        st.warning("No metrics data found for selected models")
    else:
        all_metrics = pd.concat(metrics_dfs, ignore_index=True)
        all_metrics["Station Name"] = all_metrics["station_id"].map(STATION_NAME_MAP)
        
        # Select stations to display
        unique_stations = all_metrics["Station Name"].dropna().unique().tolist()
        selected_stations = st.multiselect(
            "Select Station(s) to Compare", 
            unique_stations, 
            default=unique_stations
        )
        
        if not selected_stations:
            st.warning("Please select at least one station")
        else:
            filtered_metrics = all_metrics[all_metrics["Station Name"].isin(selected_stations)]
            
            # --- Summary metrics cards ---
            st.subheader("Summary Metrics")
            cols = st.columns(len(selected_models) if selected_models else 1)
            
            # Create a summary for each model
            for idx, model_name in enumerate(selected_models):
                if idx < len(cols):
                    model_metrics = filtered_metrics[filtered_metrics["model"] == model_name]
                    if not model_metrics.empty:
                        with cols[idx]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="margin: 0;">{model_name}</h4>
                                <p style="font-size: 0.8em; opacity: 0.8;">Average across stations</p>
                                <p style="font-size: 1.2em; margin: 10px 0 0 0;">MAE: <strong>{model_metrics['mae'].mean():.2f}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
            
            # --- MAE Bar Chart ---
            st.subheader("MAE by Station and Model")
            
            fig = px.bar(
                filtered_metrics,
                x="Station Name",
                y="mae",
                color="model",
                text="mae",
                barmode="group",
                title="Mean Absolute Error (MAE) by Station and Model",
                labels={"mae": "MAE"},
                color_discrete_map={
                    model: config["color"] for model, config in MODEL_CONFIGS.items()
                }
            )
            
            fig.update_traces(
                texttemplate='%{text:.2f}', 
                textposition='outside'
            )
            
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
            
            # --- Metrics Table ---
            st.subheader("Detailed Metrics Table")
            
            # Format and display the metrics table
            display_cols = ["model", "Station Name", "mae"]
            if "rmse" in filtered_metrics.columns:
                display_cols.append("rmse")
            if "r2" in filtered_metrics.columns:
                display_cols.append("r2")
                
            display_df = filtered_metrics[display_cols].copy()
            display_df.columns = [col.upper() if col in ["mae", "rmse", "r2"] else col.title() for col in display_df.columns]
            display_df = display_df.sort_values(by=["Station Name", "MAE"])
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Download option
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Metrics as CSV",
                csv,
                "citibike_model_metrics.csv",
                "text/csv",
                key='download-metrics'
            )
            
            # Find best model per station
            st.subheader("Best Model by Station")
            
            # Group by station and find minimum MAE
            best_models = filtered_metrics.loc[filtered_metrics.groupby("Station Name")["mae"].idxmin()]
            best_df = best_models[["Station Name", "model", "mae"]].rename(
                columns={"model": "Best Model", "mae": "MAE"}
            )
            
            st.dataframe(
                best_df,
                use_container_width=True,
                hide_index=True
            )

# ---------- PREDICTIONS VS ACTUALS TAB ----------
with tabs[1]:
    st.header("Predictions vs Actuals Analysis")
    
    # Station selection
    station_name = st.selectbox(
        "Select Station", 
        list(STATION_NAME_MAP.values()),
        key="pred_actual_station"
    )
    station_id = [k for k, v in STATION_NAME_MAP.items() if v == station_name][0]
    
    # Load predictions data
    preds_dfs = []
    for model_name in selected_models:
        if model_name in MODEL_CONFIGS:
            preds_fg = MODEL_CONFIGS[model_name]["predictions"]
            df = load_feature_group(preds_fg)
            if not df.empty:
                station_df = df[df["station_id"] == station_id].copy()
                if not station_df.empty:
                    station_df["model"] = model_name
                    preds_dfs.append(station_df)
    
    if not preds_dfs:
        st.warning(f"No prediction data found for {station_name}")
    else:
        all_preds = pd.concat(preds_dfs, ignore_index=True)
        
        # Ensure hour is datetime
        if not pd.api.types.is_datetime64_dtype(all_preds["hour"]):
            all_preds["hour"] = pd.to_datetime(all_preds["hour"], errors='coerce')
        
        # Filter by time period
        cutoff_date = datetime.now() - timedelta(days=days_to_show)
        all_preds = all_preds[all_preds["hour"] >= cutoff_date]
        
        if all_preds.empty:
            st.warning(f"No prediction data found for {station_name} within the selected time period")
        else:
            # Create prediction vs actual line plot
            st.subheader(f"Predictions vs Actuals - {station_name}")
            
            # Visualization options
            viz_option = st.radio(
                "Visualization Mode",
                ["Line Chart", "Model Comparison", "Error Analysis"],
                horizontal=True
            )
            
            if viz_option == "Line Chart":
                # Select a specific model to visualize
                model_for_viz = st.selectbox(
                    "Select Model",
                    selected_models,
                    key="line_chart_model"
                )
                
                # Filter for the selected model
                model_data = all_preds[all_preds["model"] == model_for_viz].copy()
                
                if not model_data.empty:
                    # Prepare data for plotting
                    plot_data = pd.DataFrame({
                        "hour": model_data["hour"],
                        "Actual Rides": model_data["actual_rides"],
                        "Predicted Rides": model_data["predicted_rides"]
                    })
                    
                    # Melt the data for plotting
                    plot_long = plot_data.melt(
                        id_vars="hour",
                        value_vars=["Actual Rides", "Predicted Rides"],
                        var_name="Series",
                        value_name="Rides"
                    )
                    
                    # Create the line chart
                    fig = px.line(
                        plot_long,
                        x="hour",
                        y="Rides",
                        color="Series",
                        title=f"{model_for_viz} - Actual vs Predicted Rides for {station_name}",
                        labels={"hour": "Date", "Rides": "Number of Rides"},
                        color_discrete_map={
                            "Actual Rides": "#EF3E96",  # Pink
                            "Predicted Rides": MODEL_CONFIGS[model_for_viz]["color"]
                        }
                    )
                    
                    fig.update_layout(
                        font=dict(color='white'),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(
                            title="Date",
                            tickformat="%b %d",
                            tickangle=0
                        ),
                        yaxis=dict(
                            title="Number of Rides"
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display error metrics
                    model_metrics = model_data.copy()
                    model_metrics["abs_error"] = abs(model_metrics["predicted_rides"] - model_metrics["actual_rides"])
                    model_metrics["error"] = model_metrics["predicted_rides"] - model_metrics["actual_rides"]
                    
                    mae = model_metrics["abs_error"].mean()
                    rmse = (model_metrics["error"] ** 2).mean() ** 0.5
                    
                    # Error metrics cards
                    cols = st.columns(3)
                    
                    with cols[0]:
                        st.metric("Mean Absolute Error", f"{mae:.2f}")
                    
                    with cols[1]:
                        st.metric("RMSE", f"{rmse:.2f}")
                    
                    with cols[2]:
                        over_predict = (model_metrics["error"] > 0).mean() * 100
                        st.metric("Overprediction Rate", f"{over_predict:.1f}%")
                        
            elif viz_option == "Model Comparison":
                # Create a model comparison chart
                st.subheader(f"Model Comparison - {station_name}")
                
                # Get a specific date range for comparison
                date_range = st.slider(
                    "Select Date Range",
                    min_value=all_preds["hour"].min().date(),
                    max_value=all_preds["hour"].max().date(),
                    value=(
                        (all_preds["hour"].max() - timedelta(days=7)).date(),
                        all_preds["hour"].max().date()
                    )
                )
                
                # Filter by selected date range
                start_date, end_date = date_range
                filtered_preds = all_preds[
                    (all_preds["hour"].dt.date >= start_date) & 
                    (all_preds["hour"].dt.date <= end_date)
                ]
                
                if not filtered_preds.empty:
                    # Create a comparison plot
                    fig = go.Figure()
                    
                    # Add actual rides
                    actual_data = filtered_preds.drop_duplicates(subset=["hour"])[["hour", "actual_rides"]]
                    fig.add_trace(go.Scatter(
                        x=actual_data["hour"],
                        y=actual_data["actual_rides"],
                        mode='lines',
                        name='Actual Rides',
                        line=dict(color='#EF3E96', width=3)
                    ))
                    
                    # Add each model's predictions
                    for model_name in selected_models:
                        model_data = filtered_preds[filtered_preds["model"] == model_name]
                        if not model_data.empty:
                            fig.add_trace(go.Scatter(
                                x=model_data["hour"],
                                y=model_data["predicted_rides"],
                                mode='lines',
                                name=f"{model_name}",
                                line=dict(color=MODEL_CONFIGS[model_name]["color"])
                            ))
                    
                    fig.update_layout(
                        title=f"Model Comparison for {station_name}",
                        xaxis_title="Date",
                        yaxis_title="Rides",
                        legend_title="Data Series",
                        font=dict(color='white'),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate error metrics for each model
                    model_metrics = []
                    for model_name in selected_models:
                        model_data = filtered_preds[filtered_preds["model"] == model_name]
                        if not model_data.empty:
                            mae = abs(model_data["predicted_rides"] - model_data["actual_rides"]).mean()
                            rmse = ((model_data["predicted_rides"] - model_data["actual_rides"]) ** 2).mean() ** 0.5
                            
                            model_metrics.append({
                                "Model": model_name,
                                "MAE": mae,
                                "RMSE": rmse
                            })
                    
                    if model_metrics:
                        metrics_df = pd.DataFrame(model_metrics)
                        metrics_df = metrics_df.sort_values("MAE")
                        
                        # Display metrics in a table
                        st.dataframe(
                            metrics_df,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Highlight best model
                        best_model = metrics_df.iloc[0]["Model"]
                        best_mae = metrics_df.iloc[0]["MAE"]
                        st.success(f"✅ Best model for this period: **{best_model}** with MAE of **{best_mae:.2f}**")
                    
            elif viz_option == "Error Analysis":
                # Error distribution analysis
                st.subheader("Error Distribution Analysis")
                
                # Select a model for error analysis
                model_for_error = st.selectbox(
                    "Select Model",
                    selected_models,
                    key="error_analysis_model"
                )
                
                # Filter for the selected model
                error_data = all_preds[all_preds["model"] == model_for_error].copy()
                
                if not error_data.empty:
                    # Calculate error
                    error_data["error"] = error_data["predicted_rides"] - error_data["actual_rides"]
                    error_data["abs_error"] = abs(error_data["error"])
                    error_data["percent_error"] = (error_data["abs_error"] / error_data["actual_rides"]) * 100
                    error_data["percent_error"] = error_data["percent_error"].replace([float('inf'), float('nan')], 0)
                    
                    # Calculate hourly patterns for errors
                    error_data["hour_of_day"] = error_data["hour"].dt.hour
                    error_data["day_of_week"] = error_data["hour"].dt.day_name()
                    
                    # Create tabs for different error analyses
                    error_tabs = st.tabs(["Error Distribution", "Time Patterns", "Error Over Time"])
                    
                    # Tab 1: Error Distribution
                    with error_tabs[0]:
                        # Create histogram of errors
                        fig = px.histogram(
                            error_data,
                            x="error",
                            title=f"Error Distribution for {model_for_error}",
                            labels={"error": "Prediction Error (predicted - actual)"},
                            color_discrete_sequence=[MODEL_CONFIGS[model_for_error]["color"]]
                        )
                        
                        fig.update_layout(
                            font=dict(color='white'),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Summary statistics for errors
                        error_stats = pd.DataFrame({
                            "Metric": ["Mean Error", "Mean Absolute Error", "RMSE", "Median Error",
                                     "Min Error", "Max Error", "Std Dev"],
                            "Value": [
                                error_data["error"].mean(),
                                error_data["abs_error"].mean(),
                                (error_data["error"] ** 2).mean() ** 0.5,
                                error_data["error"].median(),
                                error_data["error"].min(),
                                error_data["error"].max(),
                                error_data["error"].std()
                            ]
                        })
                        
                        st.dataframe(
                            error_stats,
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    # Tab 2: Time Patterns
                    with error_tabs[1]:
                        cols = st.columns(2)
                        
                        with cols[0]:
                            # Hourly error patterns
                            hourly_errors = error_data.groupby("hour_of_day")["abs_error"].mean().reset_index()
                            
                            fig = px.bar(
                                hourly_errors,
                                x="hour_of_day",
                                y="abs_error",
                                title="Mean Absolute Error by Hour of Day",
                                labels={"hour_of_day": "Hour of Day", "abs_error": "Mean Absolute Error"},
                                color_discrete_sequence=[MODEL_CONFIGS[model_for_error]["color"]]
                            )
                            
                            fig.update_layout(
                                font=dict(color='white'),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with cols[1]:
                            # Day of week error patterns
                            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                            daily_errors = error_data.groupby("day_of_week")["abs_error"].mean().reset_index()
                            
                            # Reorder days
                            daily_errors["day_of_week"] = pd.Categorical(
                                daily_errors["day_of_week"], 
                                categories=day_order, 
                                ordered=True
                            )
                            daily_errors = daily_errors.sort_values("day_of_week")
                            
                            fig = px.bar(
                                daily_errors,
                                x="day_of_week",
                                y="abs_error",
                                title="Mean Absolute Error by Day of Week",
                                labels={"day_of_week": "Day of Week", "abs_error": "Mean Absolute Error"},
                                color_discrete_sequence=[MODEL_CONFIGS[model_for_error]["color"]]
                            )
                            
                            fig.update_layout(
                                font=dict(color='white'),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Tab 3: Error Over Time
                    with error_tabs[2]:
                        # Track error metrics over time
                        error_data["date"] = error_data["hour"].dt.date
                        daily_metrics = error_data.groupby("date").agg({
                            "abs_error": "mean",
                            "error": ["mean", "std"]
                        }).reset_index()
                        
                        daily_metrics.columns = ["date", "MAE", "Mean Error", "Error Std Dev"]
                        
                        # Convert date back to datetime for plotting
                        daily_metrics["date"] = pd.to_datetime(daily_metrics["date"])
                        
                        # Plot MAE over time
                        fig = px.line(
                            daily_metrics,
                            x="date",
                            y="MAE",
                            title="Mean Absolute Error Over Time",
                            labels={"date": "Date", "MAE": "Mean Absolute Error"},
                            color_discrete_sequence=[MODEL_CONFIGS[model_for_error]["color"]]
                        )
                        
                        fig.update_layout(
                            font=dict(color='white'),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Plot bias (mean error) over time
                        fig = px.line(
                            daily_metrics,
                            x="date",
                            y="Mean Error",
                            title="Prediction Bias Over Time (Mean Error)",
                            labels={"date": "Date", "Mean Error": "Mean Error (+ is overprediction)"},
                            color_discrete_sequence=["#EF3E96"]
                        )
                        
                        # Add a zero line
                        fig.add_hline(
                            y=0, 
                            line_dash="dash", 
                            line_color="white",
                            annotation_text="No Bias", 
                            annotation_position="bottom right"
                        )
                        
                        fig.update_layout(
                            font=dict(color='white'),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

# ---------- FORECASTS TAB ----------
with tabs[2]:
    st.header("Future Forecasts")
    
    # Station selection for forecasts
    forecast_station = st.selectbox(
        "Select Station", 
        list(STATION_NAME_MAP.values()),
        key="forecast_station"
    )
    forecast_station_id = [k for k, v in STATION_NAME_MAP.items() if v == forecast_station][0]
    
    # Load forecast data
    forecast_dfs = []
    for model_name in selected_models:
        if model_name in MODEL_CONFIGS:
            forecast_fg = MODEL_CONFIGS[model_name]["forecast"]
            df = load_feature_group(forecast_fg)
            if not df.empty:
                station_df = df[df["station_id"] == forecast_station_id].copy()
                if not station_df.empty:
                    station_df["model"] = model_name
                    forecast_dfs.append(station_df)
    
    if not forecast_dfs:
        st.warning(f"No forecast data found for {forecast_station}")
    else:
        all_forec