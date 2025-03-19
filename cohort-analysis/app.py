import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from io import BytesIO
from dotenv import load_dotenv

# Prophet
from prophet import Prophet

# pmdarima for AutoARIMA
from pmdarima import auto_arima

# For AI commentary
from groq import Groq

# Load environment variables (for GROQ API Key)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit config
st.set_page_config(page_title="Multi-Model Financial Forecaster", layout="wide")
st.title("🔮 Multi-Model Financial Forecaster (AI-Enhanced)")

# --------------------------------------------------------------------------------
# Sidebar Inputs
st.sidebar.header("User Inputs")

uploaded_file = st.sidebar.file_uploader("📂 Upload your financial data (Excel)", type=["xlsx"])
forecast_length = st.sidebar.slider("⏳ Forecast Length (days)", min_value=30, max_value=365, value=90)

# Forecasting method selection
model_choice = st.sidebar.selectbox(
    "Select Forecasting Method",
    ["Prophet", "AutoARIMA", "Moving Average"]
)

# Additional Prophet settings
add_yearly_seasonality = st.sidebar.checkbox("Prophet: Yearly Seasonality", value=True)
add_weekly_seasonality = st.sidebar.checkbox("Prophet: Weekly Seasonality", value=True)
seasonality_mode = st.sidebar.selectbox("Prophet: Seasonality Mode", ["additive", "multiplicative"])

# Frequency map
freq_map = {
    "Daily": "D",
    "Weekly": "W",
    "Monthly": "MS"
}
freq_choice = st.sidebar.selectbox("Data Frequency", list(freq_map.keys()))

# --------------------------------------------------------------------------------
# Check for API Key
if not GROQ_API_KEY:
    st.error("🚨 GROQ API Key is missing! Please set it up in .env or Streamlit secrets.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# --------------------------------------------------------------------------------
# Main logic
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # Identify possible date columns
    possible_date_cols = df.select_dtypes(include=["datetime", "object"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found. Please check your data.")
        st.stop()

    # User picks the date & target columns
    st.markdown("### Date & Target Selection")
    if possible_date_cols:
        date_column = st.selectbox("Date Column:", possible_date_cols)
    else:
        st.warning("No date column found; generating a date range automatically.")
        date_column = None

    target_column = st.selectbox("Target (Numeric) Column:", numeric_cols)

    # Optional categorical filter
    st.markdown("### (Optional) Categorical Filter")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        selected_cat_col = st.selectbox("Choose a categorical column to filter:", [None] + list(cat_cols))
        if selected_cat_col:
            unique_vals = df[selected_cat_col].unique()
            chosen_val = st.selectbox("Choose a value:", unique_vals)
            df = df[df[selected_cat_col] == chosen_val]

    run_forecast = st.button("🚀 Generate Forecast")

    if run_forecast:
        data = df.copy()

        # Convert date column
        if date_column and date_column in data.columns:
            data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
            data = data.dropna(subset=[date_column])
            data = data.sort_values(by=date_column)
            data.rename(columns={date_column: "ds"}, inplace=True)
        else:
            # Auto-generate date
            data = data.reset_index(drop=True)
            data["ds"] = pd.date_range("2020-01-01", periods=len(data), freq=freq_map[freq_choice])

        # Clean target
        data.rename(columns={target_column: "y"}, inplace=True)
        data = data.dropna(subset=["y"])

        # Prepare forecast horizon
        future_dates = pd.date_range(
            start=data["ds"].iloc[-1] + pd.Timedelta(days=1),
            periods=forecast_length,
            freq=freq_map[freq_choice]
        )

        forecast_df = None

        # --------------------------------------------------
        # 1) Prophet
        if model_choice == "Prophet":
            m = Prophet(
                yearly_seasonality=add_yearly_seasonality,
                weekly_seasonality=add_weekly_seasonality,
                seasonality_mode=seasonality_mode
            )
            m.fit(data[["ds", "y"]])
            future = m.make_future_dataframe(periods=forecast_length, freq=freq_map[freq_choice])
            forecast_df = m.predict(future)

            st.subheader("Prophet Forecast Results")
            st.dataframe(forecast_df.tail())

            fig1 = m.plot(forecast_df)
            st.pyplot(fig1)

            fig2 = m.plot_components(forecast_df)
            st.pyplot(fig2)

        # --------------------------------------------------
        # 2) AutoARIMA
        elif model_choice == "AutoARIMA":
            # For simplicity, convert ds to a DateTimeIndex
            ts = data.set_index("ds")["y"].asfreq(freq_map[freq_choice])
            ts = ts.fillna(method="ffill")  # fill missing if needed

            # Train AutoARIMA
            model_arima = auto_arima(
                ts,
                seasonal=True,
                m=12 if freq_choice == "Monthly" else 7,  # example: 12 for monthly, 7 for weekly
                trace=False,
                error_action="ignore",
                suppress_warnings=True
            )
            # Predict future
            forecast_vals = model_arima.predict(n_periods=forecast_length, return_conf_int=True)
            fc_series = forecast_vals[0]
            conf_int = forecast_vals[1]

            # Prepare forecast df
            forecast_idx = pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=forecast_length, freq=freq_map[freq_choice])
            forecast_df = pd.DataFrame({
                "ds": forecast_idx,
                "yhat": fc_series,
                "yhat_lower": conf_int[:, 0],
                "yhat_upper": conf_int[:, 1]
            })

            st.subheader("AutoARIMA Forecast Results")
            st.dataframe(forecast_df.tail())

            # Simple plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(ts.index, ts, label="Historical")
            ax.plot(forecast_idx, fc_series, label="Forecast", color="green")
            ax.fill_between(forecast_idx, conf_int[:, 0], conf_int[:, 1], color="k", alpha=0.1)
            ax.legend()
            ax.set_title("AutoARIMA Forecast")
            st.pyplot(fig)

        # --------------------------------------------------
        # 3) Moving Average Baseline
        elif model_choice == "Moving Average":
            # Basic moving average forecast
            ts = data.set_index("ds")["y"].asfreq(freq_map[freq_choice])
            ts = ts.fillna(method="ffill") 

            # We'll just take the last known average of a certain window
            window_size = 7 if freq_choice == "Daily" else 4
            moving_avg = ts.rolling(window=window_size).mean().iloc[-1]

            # Generate forecast
            forecast_idx = pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=forecast_length, freq=freq_map[freq_choice])
            forecast_series = np.full(len(forecast_idx), moving_avg)

            # Construct forecast dataframe
            forecast_df = pd.DataFrame({
                "ds": forecast_idx,
                "yhat": forecast_series
            })
            # For simplicity, no confidence intervals are defined here
            forecast_df["yhat_lower"] = forecast_df["yhat"]
            forecast_df["yhat_upper"] = forecast_df["yhat"]

            st.subheader("Moving Average Forecast Results")
            st.dataframe(forecast_df.tail())

            # Simple plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(ts.index, ts, label="Historical")
            ax.plot(forecast_idx, forecast_series, label="Moving Average Forecast", color="green")
            ax.legend()
            ax.set_title("Moving Average Baseline Forecast")
            st.pyplot(fig)

        # --------------------------------------------------
        # Download forecast results
        if forecast_df is not None:
            forecast_buffer = BytesIO()
            forecast_df.to_excel(forecast_buffer, index=False)
            st.download_button(
                label="📥 Download Forecast Data",
                data=forecast_buffer.getvalue(),
                file_name="forecast_results.xlsx"
            )

            # Summaries & AI Integration
            st.subheader("🤖 AI-Generated Insights")

            # Use only the new forecast period (last N entries)
            if len(forecast_df) > forecast_length:
                forecast_period = forecast_df.tail(forecast_length)
            else:
                forecast_period = forecast_df

            # Basic summary
            lower_bound = forecast_period["yhat_lower"].min() if "yhat_lower" in forecast_period.columns else forecast_period["yhat"].min()
            upper_bound = forecast_period["yhat_upper"].max() if "yhat_upper" in forecast_period.columns else forecast_period["yhat"].max()

            forecast_summary = f"""
Forecast Method: {model_choice}
Forecast Period: {forecast_period['ds'].iloc[0].strftime('%Y-%m-%d')} - {forecast_period['ds'].iloc[-1].strftime('%Y-%m-%d')}
Average Forecast: ${forecast_period['yhat'].mean():,.2f}
Range: ${lower_bound:,.2f} - ${upper_bound:,.2f}
"""

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert financial analyst providing insights on forecasts."
                },
                {
                    "role": "user",
                    "content": (
                        "Here is the summary of the forecast:\n\n"
                        f"{forecast_summary}\n\n"
                        "Please provide a detailed analysis including:\n"
                        "1. Key trends.\n"
                        "2. Potential risks.\n"
                        "3. Strategic recommendations."
                    )
                }
            ]

            try:
                response = client.chat.completions.create(
                    messages=messages,
                    model="llama3-8b-8192",
                )
                ai_commentary = response.choices[0].message.content
            except Exception as e:
                ai_commentary = f"Error retrieving AI commentary: {str(e)}"

            st.write(ai_commentary)
else:
    st.info("Please upload an Excel file to begin.")
