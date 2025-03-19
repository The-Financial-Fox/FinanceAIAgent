import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import xgboost as xgb
from groq import Groq

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit UI
st.title("📈 AI-Driven Revenue Forecasting Agent")
st.write("Upload revenue data and get AI-powered insights!")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your revenue data (CSV format)", type=["csv"])

if uploaded_file:
    # Read CSV file
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
    df = df.sort_values(by='Date')
    
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())
    
    # User selects forecast period
    forecast_period = st.slider("Select Forecast Period (months)", 3, 24, 6)
    
    # ARIMA Forecast
    def arima_forecast(df, periods):
        model = ARIMA(df['Revenue'], order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)
        return forecast
    
    # Prophet Forecast
    def prophet_forecast(df, periods):
        df_prophet = df.rename(columns={'Date': 'ds', 'Revenue': 'y'})
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)
        return forecast[['ds', 'yhat']]
    
    # XGBoost Forecast
    def xgboost_forecast(df, periods):
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        X = df[['Month', 'Year']]
        y = df['Revenue']
        model = xgb.XGBRegressor(objective='reg:squarederror')
        model.fit(X, y)
        future_dates = pd.date_range(start=df['Date'].max(), periods=periods, freq='M')
        future_df = pd.DataFrame({'Month': future_dates.month, 'Year': future_dates.year})
        forecast = model.predict(future_df)
        return future_dates, forecast
    
    # Generate forecasts
    arima_pred = arima_forecast(df, forecast_period)
    prophet_pred = prophet_forecast(df, forecast_period)
    future_dates, xgb_pred = xgboost_forecast(df, forecast_period)
    
    # Plot results
    st.subheader("📊 Forecast Results")
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Revenue'], label='Actual Revenue', marker='o')
    plt.plot(prophet_pred['ds'].tail(forecast_period), prophet_pred['yhat'].tail(forecast_period), label='Prophet Forecast', linestyle='dashed')
    plt.plot(future_dates, xgb_pred, label='XGBoost Forecast', linestyle='dotted')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.title('Revenue Forecast Comparison')
    st.pyplot(plt)
    
    # AI Summary Preparation
    ai_summary = f"""
    📌 **Revenue Forecast Summary**:
    - Forecast Period: {forecast_period} months
    - ARIMA Forecast: {list(arima_pred)}
    - Prophet Forecast: {list(prophet_pred['yhat'].tail(forecast_period))}
    - XGBoost Forecast: {list(xgb_pred)}
    """
    
    # AI Agent for Insights
    st.subheader("🤖 AI Insights - Revenue Growth Levers")
    user_prompt = st.text_area("📝 Ask AI about revenue trends:", "Analyze forecast data and suggest growth strategies.")
    
    if st.button("🚀 Generate AI Insights"):
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI financial analyst providing revenue insights."},
                {"role": "user", "content": f"The forecasted revenue data is:
                {ai_summary}
                {user_prompt}"}
            ],
            model="llama3-8b-8192",
        )
        ai_commentary = response.choices[0].message.content
        
        # Display AI commentary
        st.subheader("💡 AI-Generated Revenue Insights")
        st.write(ai_commentary)
