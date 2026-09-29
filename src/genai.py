import os
import joblib
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables
load_dotenv()

# Create OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Load trained ML model
model = joblib.load("models/hiring_model.pkl")


def create_hiring_prompt(candidate, prediction, probability):

    prompt = f"""
You are an AI hiring assistant.

Analyze the following candidate information:

Candidate details:
{candidate}

Machine learning prediction:
Shortlisted: {prediction}
Shortlist probability: {probability:.2%}

Provide:
1. Candidate strengths
2. Candidate weaknesses
3. Hiring analysis
4. Recommendations for improvement

Important:
- Do not change or override the machine learning prediction.
- Do not claim that you know the exact reason for the model's decision.
- Base your explanation only on the candidate information provided.
- Your role is to explain the prediction and provide recommendations.
"""

    return prompt


def generate_ai_analysis(prompt):

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return response.output_text


# --------------------------------
# Candidate information
# --------------------------------

candidate = {
    "age": 24,
    "experience": 2,
    "cgpa": 8.7,
    "coding_score": 88,
    "python_score": 90,
    "sql_score": 78,
    "projects": 3,
    "internships": 1,
    "communication_score": 85
}


# --------------------------------
# Convert candidate into DataFrame
# --------------------------------

candidate_df = pd.DataFrame([candidate])


# --------------------------------
# ML prediction
# --------------------------------

prediction = model.predict(candidate_df)[0]

probability = model.predict_proba(candidate_df)[0][1]


print("ML Prediction:", prediction)
print("Shortlist Probability:", f"{probability:.2%}")


# --------------------------------
# Generate GenAI prompt
# --------------------------------

prompt = create_hiring_prompt(
    candidate,
    prediction,
    probability
)


# --------------------------------
# Generate AI analysis
# --------------------------------

ai_response = generate_ai_analysis(prompt)


print("\n===== AI HIRING ANALYSIS =====")
print(ai_response)