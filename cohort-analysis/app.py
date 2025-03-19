import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os
from groq import Groq
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
os.environ["GROQ_API_KEY"] = "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"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit App UI
st.title("🔮 Financial Forecaster - AI-Powered Predictions")
st.write("Upload an Excel file, select a column for forecasting, and get AI-generated financial insights!")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your financial data (Excel format)", type=["xlsx"])

if uploaded_file:
    # Read the uploaded Excel file
    df = pd.read_excel(uploaded_file)

    # Display data preview
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # Select column for forecasting
    target_column = st.selectbox("📌 Select the column to forecast:", df.columns)
    
    # Select categorical column for filtering (if applicable)
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        selected_category_column = st.selectbox("🎯 Select a categorical column to filter (Optional):", [None] + list(categorical_columns))
        if selected_category_column:
            unique_values = df[selected_category_column].unique()
            selected_value = st.selectbox(f"🔍 Select value from {selected_category_column}:", unique_values)
            df = df[df[selected_category_column] == selected_value]

    # User input for forecast length
    forecast_length = st.slider("⏳ Select the forecast length (days):", min_value=30, max_value=365, value=180)

    if st.button("🚀 Generate Forecast"):
        # Prepare data for Prophet
        forecast_data = df.copy()
        forecast_data = forecast_data.rename(columns={target_column: "y"})
        forecast_data['ds'] = pd.date_range(start='2022-01-01', periods=len(df), freq='D')

        # Train Prophet Model
        model = Prophet(yearly_seasonality=True)
        model.fit(forecast_data)

        # Create future dates for prediction
        future = model.make_future_dataframe(periods=forecast_length)
        forecast = model.predict(future)

        # Display Forecast Data
        st.subheader("🔍 Forecasted Data Preview")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        # Allow user to download forecast results
        forecast_file_path = "financial_forecast_results.xlsx"
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_excel(forecast_file_path, index=False)
        st.download_button(label="📥 Download Forecast Data", data=open(forecast_file_path, "rb"), file_name="forecast_results.xlsx")

        # Plot Forecast
        st.subheader("📈 Forecast Visualization")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        # Forecast Components
        st.subheader("📊 Forecast Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

        # Prepare summary for AI
        forecast_summary = f"""
        Financial Forecast Summary:
        - Forecasted Period: {forecast['ds'].iloc[-forecast_length].strftime('%Y-%m-%d')} to {forecast['ds'].iloc[-1].strftime('%Y-%m-%d')}
        - Expected Range:
          - Lower Bound: ${forecast['yhat_lower'].iloc[-forecast_length:].min():,.2f}
          - Upper Bound: ${forecast['yhat_upper'].iloc[-forecast_length:].max():,.2f}
        - Average Forecasted Value: ${forecast['yhat'].iloc[-forecast_length:].mean():,.2f}
        """

        # AI Commentary Section
        st.subheader("🤖 AI-Generated Strategic Insights")

        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert FP&A analyst providing insights on financial forecasts."},
                {"role": "user", "content": f"The financial forecast is summarized below:\n{forecast_summary}\nPlease provide insights, identify trends, and give strategic recommendations."}
            ],
            model="llama3-8b-8192",
        )

        ai_commentary = response.choices[0].message.content

        # Display AI Commentary
        st.subheader("💡 AI-Powered Forecast Insights")
        st.write(ai_commentary)
