import streamlit as st
import pandas as pd
import os
from groq import Groq
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# **🎨 Streamlit UI Styling**
st.set_page_config(page_title="Financial Data Storyteller AI", page_icon="📊", layout="wide")

st.markdown("""
    <style>
        .title { text-align: center; font-size: 36px; font-weight: bold; color: #2E0249; }
        .subtitle { text-align: center; font-size: 20px; color: #4A0072; }
        .stButton>button { width: 100%; background-color: #2E0249; color: white; font-size: 16px; font-weight: bold; }
        .stFileUploader { text-align: center; }
        .story-container { padding: 15px; border-radius: 10px; margin: 10px 0; background-color: #EDE7F6; }
        .story-title { font-size: 20px; font-weight: bold; color: #4A0072; }
        .story-desc { font-size: 16px; color: #2E0249; }
    </style>
""", unsafe_allow_html=True)

# **📢 Title & Description**
st.markdown('<h1 class="title">📊 Financial Data Storyteller AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload your financial data, and AI will craft a compelling data story with actionable insights.</p>', unsafe_allow_html=True)

# **📂 Upload Financial Data**
st.subheader("📥 Upload Your Financial Data (Excel)")
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    st.subheader("🔍 Data Preview")
    st.dataframe(df)

    # Convert dataframe to a summary for AI
    data_summary = df.describe(include='all').to_string()

    # **User Input: Key Focus Areas**
    st.subheader("🎯 What is the key message of your presentation?")
    user_input = st.text_area("💡 Describe the main financial insights or business questions you want to highlight.")

    if st.button("🚀 Generate Data Storytelling Strategy"):
        client = Groq(api_key=GROQ_API_KEY)

        # AI Prompt for Storytelling
        prompt = f"""
        You are an expert in financial storytelling and executive presentations.
        The user has uploaded financial data, and your task is to analyze it and craft a compelling story.
        Your output should follow this structure:
        1️⃣ **Overall Data Insights** – Key trends and findings from the uploaded data.
        2️⃣ **Strategic Message** – What is the most important insight for senior stakeholders?
        3️⃣ **Slide Narrative** – A structured sequence of slides with content for a CFO presentation.
        4️⃣ **Actionable Takeaways** – Clear recommendations based on the data.

        Data Summary:
        {data_summary}

        Key User Input: {user_input}
        """

        response = client.chat.completions.create(
            messages=[{"role": "system", "content": "You are a finance storytelling expert."},
                      {"role": "user", "content": prompt}],
            model="llama3-8b-8192",
        )

        story_response = response.choices[0].message.content

        # **Display AI Storytelling Strategy**
        st.subheader("📖 AI-Generated Financial Storytelling Strategy")
        st.markdown('<div class="story-container">', unsafe_allow_html=True)
        st.write(story_response)
        st.markdown('</div>', unsafe_allow_html=True)
