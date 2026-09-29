import os

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from openai import OpenAI


# ==========================================
# Configuration
# ==========================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

model = joblib.load("models/hiring_model.pkl")


# ==========================================
# GenAI Prompt
# ==========================================

def create_hiring_prompt(
    candidate,
    prediction,
    probability,
    feature_importance
):

    prompt = f"""
You are an AI hiring assistant.

Analyze the following candidate information.

Candidate details:
{candidate}

Machine learning prediction:
Shortlisted: {prediction}
Shortlist probability: {probability:.2%}

Random Forest feature importance:
{feature_importance}

Generate the response using exactly these four sections:

## 💪 Candidate Strengths

Give 3-5 concise bullet points.

## ⚠️ Candidate Weaknesses

Give 2-4 concise bullet points.

## 📊 Hiring Analysis

Explain the candidate profile using the candidate data and ML output.

Do not claim that feature importance proves causation.

## 🎯 Recommendations

Give 3-5 practical recommendations.

Important rules:

- Do not change or override the ML prediction.
- Do not invent candidate information.
- Do not claim that feature importance proves causation.
- Treat feature importance only as a model-derived signal.
- Clearly distinguish the ML model output from your analysis.
- Keep the response concise and professional.
"""

    return prompt


# ==========================================
# Generate AI Analysis
# ==========================================

def generate_ai_analysis(prompt):

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return response.output_text


# ==========================================
# Streamlit Configuration
# ==========================================

st.set_page_config(
    page_title="Smart Hiring AI",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# Header
# ==========================================

st.title("🤖 Smart Hiring AI")

st.write(
    "AI-powered candidate screening and hiring analysis"
)

st.divider()


# ==========================================
# Candidate Information
# ==========================================

st.header("👤 Candidate Information")

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=60,
        value=24
    )

    experience = st.number_input(
        "Experience (years)",
        min_value=0,
        max_value=30,
        value=2
    )

    cgpa = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=8.7,
        step=0.1
    )


with col2:

    coding_score = st.slider(
        "Coding Score",
        0,
        100,
        88
    )

    python_score = st.slider(
        "Python Score",
        0,
        100,
        90
    )

    sql_score = st.slider(
        "SQL Score",
        0,
        100,
        78
    )


with col3:

    projects = st.number_input(
        "Projects",
        min_value=0,
        max_value=20,
        value=3
    )

    internships = st.number_input(
        "Internships",
        min_value=0,
        max_value=10,
        value=1
    )

    communication_score = st.slider(
        "Communication Score",
        0,
        100,
        85
    )


st.divider()


# ==========================================
# Analyze Candidate
# ==========================================

if st.button(
    "🔍 Analyze Candidate",
    type="primary"
):

    # --------------------------------------
    # Candidate data
    # --------------------------------------

    candidate = {
        "age": age,
        "experience": experience,
        "cgpa": cgpa,
        "coding_score": coding_score,
        "python_score": python_score,
        "sql_score": sql_score,
        "projects": projects,
        "internships": internships,
        "communication_score": communication_score
    }


    # --------------------------------------
    # DataFrame
    # --------------------------------------

    candidate_df = pd.DataFrame(
        [candidate]
    )


    # --------------------------------------
    # ML Prediction
    # --------------------------------------

    prediction = model.predict(
        candidate_df
    )[0]

    probability = model.predict_proba(
        candidate_df
    )[0][1]


    # --------------------------------------
    # Feature Importance
    # --------------------------------------

    importance_df = pd.DataFrame({
        "Feature": candidate_df.columns,
        "Importance": model.feature_importances_
    })

    importance_df["Importance"] = (
        importance_df["Importance"].astype(float)
    )

    feature_importance = dict(
        zip(
            importance_df["Feature"],
            importance_df["Importance"].round(4)
        )
    )


    # ======================================
    # ML RESULT
    # ======================================

    st.header("📊 ML Prediction")

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        if prediction == 1:

            st.success(
                "✅ Candidate Shortlisted"
            )

        else:

            st.error(
                "❌ Candidate Not Shortlisted"
            )


    with result_col2:

        st.metric(
            "Shortlist Probability",
            f"{probability:.2%}"
        )


    # ======================================
    # Candidate Summary
    # ======================================

    st.subheader("Candidate Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)


    with summary_col1:

        st.metric(
            "CGPA",
            cgpa
        )

        st.metric(
            "Coding Score",
            coding_score
        )


    with summary_col2:

        st.metric(
            "Python Score",
            python_score
        )

        st.metric(
            "SQL Score",
            sql_score
        )


    with summary_col3:

        st.metric(
            "Projects",
            projects
        )

        st.metric(
            "Experience",
            f"{experience} years"
        )


    # ======================================
    # Feature Importance
    # ======================================

    st.divider()

    st.header("🧠 Model Feature Importance")


    # Sort for better horizontal chart
    importance_df = importance_df.sort_values(
        "Importance",
        ascending=True
    )


    # Create Matplotlib chart
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )


    ax.set_xlabel(
        "Importance"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "Random Forest Feature Importance"
    )


    # Add values to bars
    for index, value in enumerate(
        importance_df["Importance"]
    ):

        ax.text(
            value,
            index,
            f" {value:.3f}",
            va="center"
        )


    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # ======================================
    # GenAI Analysis
    # ======================================

    st.divider()

    st.header("🤖 AI Hiring Analysis")


    with st.spinner(
        "Generating AI hiring analysis..."
    ):

        prompt = create_hiring_prompt(
            candidate,
            prediction,
            probability,
            feature_importance
        )

        ai_response = generate_ai_analysis(
            prompt
        )


    # Display AI response
    st.markdown(
        ai_response
    )