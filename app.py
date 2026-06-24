"""
AI-Hiring Assistant — Phase 6: Emotion Analysis, Fraud Detection & Final Decision Engine
Run: streamlit run app.py
"""

# ── Imports ──────────────────────────────────────────────────────────────
import streamlit as st
import pdfplumber
from modules.resume_parser import extract_resume_text
from PIL import Image
import numpy as np
# MUST be first Streamlit command
st.set_page_config(
    page_title="AI-Hiring Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os, sys, sqlite3, warnings
from datetime import datetime
from collections import Counter

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# Ensure root path access
sys.path.insert(0, os.path.dirname(__file__))

# ── Module Imports ───────────────────────────────────────────────────────
from modules.emotion_analysis import (
    analyze_image, analyze_video_frames,
    calculate_behavioral_scores, get_emotion_color,
    EMOTIONS, CV2_AVAILABLE, DEEPFACE_AVAILABLE,
)

from modules.fraud_detection import (
    calculate_fraud_score, analyze_dataset_fraud,
    get_fraud_summary,
)

from modules.decision_engine import (
    calculate_final_score, batch_score_candidates,
    get_executive_summary, generate_reasoning,
    RECOMMENDATION_COLORS,
)

from modules.report_generator import (
    generate_hiring_csv, generate_summary_csv,
    generate_hiring_pdf, generate_candidate_profile_txt,
)

# ── Paths ────────────────────────────────────────────────────────────────
DB_PATH  = os.path.join("database", "hiring_system.db")
OUT_DIR  = "outputs"
DATA_CSV = os.path.join("data", "candidate_resume_dataset.csv")

# ── Init DB ──────────────────────────────────────────────────────────────
def init_db():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_name TEXT,
            role TEXT,
            experience_years REAL,
            resume_score REAL DEFAULT 0,
            interview_score REAL DEFAULT 0,
            voice_score REAL DEFAULT 0,
            behavioral_score REAL DEFAULT 0,
            confidence_score REAL DEFAULT 0,
            engagement_score REAL DEFAULT 0,
            stability_score REAL DEFAULT 0,
            fraud_score REAL DEFAULT 0,
            fraud_flags TEXT DEFAULT '',
            risk_level TEXT DEFAULT 'Low',
            emotion_dominant TEXT DEFAULT 'neutral',
            emotion_json TEXT DEFAULT '{}',
            final_score REAL DEFAULT 0,
            recommendation TEXT DEFAULT '',
            reasoning TEXT DEFAULT '',
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()

init_db()
os.makedirs(OUT_DIR, exist_ok=True)

# ── UI HEADER ────────────────────────────────────────────────────────────
st.title("🚀 AI Hiring Assistant Dashboard")
st.markdown("### End-to-End AI Recruitment System")

st.markdown("---")

# ── Sidebar Navigation ───────────────────────────────────────────────────
PAGES = [
    "📄 Resume Analysis",
    "🎯 Skill Gap Analysis",
    "🏆 Top Candidates",
    "📈 Hiring Funnel",
    "🎤 Interview Analytics",
    "📉 Hiring Trends",
    "🧠 Emotion Analysis",
    "🚨 Fraud Detection",
    "📊 Final Hiring Score",
    "🤖 Decision Engine"
]

page = st.sidebar.selectbox("Navigate", PAGES)

# ── Helper Functions ─────────────────────────────────────────────────────
def gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        gauge={"axis": {"range": [0, 100]}}
    ))
    return fig

# ── Page 1: Emotion Analysis ─────────────────────────────────────────────
if page == "📄 Resume Analysis":

    st.header("Resume Analysis")

    uploaded_resume = st.file_uploader(
        "Upload Resume",
        type=["pdf"]
    )

    if uploaded_resume:

        resume_text = extract_resume_text(
            uploaded_resume
        )

        st.subheader("Extracted Resume Text")

        st.text_area(
            "Resume Content",
            resume_text,
            height=300
        )
if page == "🧠 Emotion Analysis":

    st.header("Emotion Analysis")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        st.image(uploaded_file)

        try:

            from PIL import Image
            import numpy as np

            image = Image.open(uploaded_file)
            image_np = np.array(image)

            result = analyze_image(image_np)

            st.success("Emotion Analysis Complete")

            st.write(result)

        except Exception as e:

            st.error(f"Error: {e}")
# ── Page 2: Fraud Detection ──────────────────────────────────────────────
elif page == "🚨 Fraud Detection":
    st.header("Fraud Detection")

    if os.path.exists(DATA_CSV):
        df = pd.read_csv(DATA_CSV)
        fraud_df = analyze_dataset_fraud(df)

        st.dataframe(fraud_df)

    else:
        st.warning("Dataset not found")

# ── Page 3: Final Score ──────────────────────────────────────────────────
elif page == "📊 Final Hiring Score":
    st.header("Final Candidate Scoring")

    sample = {
        "resume": 80,
        "interview": 75,
        "behavior": 70,
        "fraud": 10
    }

    score_result = calculate_final_score(sample)

    st.plotly_chart(gauge_chart(score_result["final_score"], "Final Score"))

# ── Page 4: Decision Engine ──────────────────────────────────────────────
elif page == "🤖 Decision Engine":
    st.header("AI Hiring Decision")

    sample = {
        "resume": 85,
        "interview": 80,
        "behavior": 75,
        "fraud": 5
    }

    score_result = calculate_final_score(sample)
    reasoning = generate_reasoning(sample)

    st.metric("Final Score", score_result["final_score"])
    st.write("### AI Reasoning")
    st.write(reasoning)

# ── Footer ───────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("AI Hiring Assistant © 2026")