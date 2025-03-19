import streamlit as st
import pandas as pd
import os
import io
from groq import Groq
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit App UI
st.set_page_config(page_title="Excel Formula Generator", page_icon="📊", layout="wide")
st.title("📊 Excel Formula Generator – AI-Powered Financial Modeling")
st.write("Describe the calculation you need, and AI will generate the best Excel formula for you!")

# User Input for Formula Description
user_prompt = st.text_area("📝 Describe the Excel formula you need (e.g., 'Calculate the CAGR for a 5-year period', 'Find the top 5 highest sales values in a range', etc.): ")

if st.button("🚀 Generate Excel Formula"):
    if user_prompt:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant that generates Excel formulas for financial modeling and complex calculations. "
                        "Users will describe what they need, and you will provide the most efficient Excel formula for it. "
                        "Also, provide an explanation of how the formula works and an example use case."
                    ),
                },
                {"role": "user", "content": f"Generate an Excel formula for this task: {user_prompt}. Explain how it works and provide an example."}
            ],
            model="llama3-8b-8192",
        )

        ai_response = response.choices[0].message.content

        # **Display AI-Generated Formula**
        st.subheader("📊 AI-Generated Excel Formula")
        st.write(ai_response)

        # **Generate Sample Excel File with the Formula**
        st.subheader("📥 Generate Sample Excel File")
        if st.button("📂 Create Example Excel File"):
            try:
                # Create a simple sample dataset
                data = {
                    "Year": [2020, 2021, 2022, 2023, 2024],
                    "Revenue": [100000, 120000, 150000, 180000, 210000],
                    "Formula Applied": ["=CAGR(A2:A6, B2:B6)"] * 5  # Placeholder Formula
                }

                df = pd.DataFrame(data)

                # Save to Excel
                file_path = "generated_excel_formula.xlsx"
                with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False, sheet_name="Formula Example")

                    # Add formula in Excel
                    worksheet = writer.sheets["Formula Example"]
                    worksheet.write_formula("D2", "=POWER(B6/B2, 1/(A6-A2))-1")  # Example formula for CAGR

                st.download_button(label="📥 Download Excel File", data=open(file_path, "rb"), file_name="generated_excel_formula.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            except Exception as e:
                st.error(f"⚠️ Error generating Excel file: {e}")
