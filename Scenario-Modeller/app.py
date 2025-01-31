import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from groq import Groq
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit App UI
st.set_page_config(page_title="Scenario Planning AI", page_icon="📊", layout="wide")
st.title("📊 Scenario Planning AI – Simulate Financial Scenarios")
st.write("Upload financial data and enter a scenario prompt to simulate different projections!")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your dataset (Excel format)", type=["xlsx"])

if uploaded_file:
    # Read the Excel file
    df = pd.read_excel(uploaded_file)

    # Check for required columns
    required_columns = ["Category", "Base Forecast"]
    if not all(col in df.columns for col in required_columns):
        st.error("⚠️ The uploaded file must contain 'Category' and 'Base Forecast' columns!")
        st.stop()

    # Scenario Input
    scenario_prompt = st.text_area("📝 Enter a financial scenario (e.g., 'Revenue drops 10%', 'Costs increase by 5%'):")

    if st.button("🚀 Generate Scenarios"):
        # Generate Different Scenario Projections
        df["Optimistic"] = df["Base Forecast"] * np.random.uniform(1.1, 1.3, len(df))
        df["Pessimistic"] = df["Base Forecast"] * np.random.uniform(0.7, 0.9, len(df))
        df["Worst Case"] = df["Base Forecast"] * np.random.uniform(0.5, 0.7, len(df))

        # Display scenario data
        st.subheader("📊 Scenario-Based Projections")
        st.dataframe(df)

        # Plot Scenario Analysis
        fig_scenarios = px.bar(
            df,
            x="Category",
            y=["Base Forecast", "Optimistic", "Pessimistic", "Worst Case"],
            title="📉 Scenario Planning: Financial Projections",
            barmode="group",
            text_auto=".2s",
        )
        st.plotly_chart(fig_scenarios)

        # AI Section
        st.subheader("🤖 AI-Powered Scenario Analysis")

        # AI Summary of Scenario Data
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI financial analyst providing scenario planning insights based on different projections."},
                {"role": "user", "content": f"Here are the scenario projections:\n{df.to_string()}\nScenario: {scenario_prompt}\nWhat are the key insights and recommendations?"}
            ],
            model="llama3-8b-8192",
        )

        st.write(response.choices[0].message.content)

        # AI Chat - Users Can Ask Questions
        st.subheader("🗣️ Chat with AI About Scenario Planning")

        user_query = st.text_input("🔍 Ask the AI about financial scenarios:")
        if user_query:
            chat_response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an AI financial strategist helping users understand scenario-based financial modeling."},
                    {"role": "user", "content": f"Scenario Data:\n{df.to_string()}\n{user_query}"}
                ],
                model="llama3-8b-8192",
            )
            st.write(chat_response.choices[0].message.content)
