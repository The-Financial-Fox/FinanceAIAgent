import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from groq import Groq
from dotenv import load_dotenv
from operator import attrgetter

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit App UI
st.title("🤖 FP&A AI Agent - SaaS Cohort Analysis")
st.write("Upload an Excel file, analyze retention rates, and get AI-generated FP&A insights!")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your cohort data (Excel format)", type=["xlsx"])

if uploaded_file:
    # Read the Excel file
    sales_data = pd.read_excel(uploaded_file)

    # Convert Date column to datetime
    sales_data['Date'] = pd.to_datetime(sales_data['Date'])

    # Extract the first purchase month for each customer
    sales_data['CohortMonth'] = sales_data.groupby('Customer_ID')['Date'].transform('min').dt.to_period('M')

    # Calculate month difference between the purchase date and the cohort month
    sales_data['PurchaseMonth'] = sales_data['Date'].dt.to_period('M')
    sales_data['CohortIndex'] = (sales_data['PurchaseMonth'] - sales_data['CohortMonth']).apply(attrgetter('n'))

    # Create a pivot table for cohort analysis
    cohort_counts = sales_data.pivot_table(index='CohortMonth', columns='CohortIndex', values='Customer_ID', aggfunc='nunique')

    # Calculate retention rates
    cohort_sizes = cohort_counts.iloc[:, 0]
    retention_rate = cohort_counts.divide(cohort_sizes, axis=0)

    # Display data preview
    st.subheader("📊 Data Preview")
    st.dataframe(sales_data.head())

    # Plot retention rate heatmap
    st.subheader("🔥 Retention Rate Heatmap")
    plt.figure(figsize=(16, 9))
    sns.heatmap(retention_rate, annot=True, fmt=".0%", cmap="YlGnBu", linewidths=0.5)
    plt.title('Cohort Analysis - Retention Rate', fontsize=16)
    plt.xlabel('Months Since First Purchase', fontsize=12)
    plt.ylabel('Cohort Month', fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)

    # Prepare cohort summary for AI
    cohort_summary = f"""
    📌 **Cohort Analysis Summary**:
    - Number of Cohorts: {len(cohort_counts)}
    - Retention Rate Breakdown:
    {retention_rate.to_string()}
    """

    # AI Agent Section
    st.subheader("🤖 AI Agent - FP&A Commentary")

    # User Prompt Input
    user_prompt = st.text_area("📝 Enter your question for the AI:", "Analyze the cohort retention data and provide key FP&A insights.")

    if st.button("🚀 Generate AI Commentary"):
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI-powered FP&A analyst providing financial insights."},
                {"role": "user", "content": f"The cohort retention analysis is summarized below:\n{cohort_summary}\n{user_prompt}"}
            ],
            model="llama-3.1-8b-instant",
        )

        ai_commentary = response.choices[0].message.content

        # Display AI commentary
        st.subheader("💡 AI-Generated FP&A Insights")
        st.write(ai_commentary)
