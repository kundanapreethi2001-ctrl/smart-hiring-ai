# 🤖 Smart Hiring AI

An AI-powered candidate screening system that combines Machine Learning and Generative AI to assist with candidate evaluation.

## 🚀 Overview

Smart Hiring AI predicts whether a candidate should be shortlisted using a trained Random Forest classification model.

The application then uses Generative AI to explain the model output by analyzing the candidate's strengths, weaknesses, and areas for improvement.

### Architecture

Candidate Information
        ↓
Data Processing
        ↓
Random Forest Model
        ↓
Prediction + Shortlist Probability
        ↓
Feature Importance
        ↓
Generative AI
        ↓
Hiring Analysis
        ↓
Streamlit Web Application

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- OpenAI API
- Streamlit
- MySQL
- Joblib
- Git & GitHub

## 📊 Machine Learning

The project uses a Random Forest classifier to predict candidate shortlisting.

Candidate features include:

- Age
- Experience
- CGPA
- Coding score
- Python score
- SQL score
- Number of projects
- Number of internships
- Communication score

### Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Feature Importance

The project also compares Logistic Regression and Random Forest models.

## 🤖 Generative AI

After the ML model generates a prediction, the Generative AI layer provides:

- Candidate strengths
- Candidate weaknesses
- Hiring analysis
- Recommendations for improvement

The Generative AI layer does not make or override the machine learning prediction.

## 🌐 Streamlit Application

The Streamlit application allows users to enter candidate information and receive:

- Shortlisting prediction
- Shortlist probability
- Model feature importance
- AI-generated candidate analysis

## 📁 Project Structure

```text
smart-hiring-ai/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── data/
│   └── candidates.csv
│
├── models/
│   └── hiring_model.pkl
│
├── sql/
│   └── candidate_analysis.sql
│
└── src/
    ├── explore_data.py
    ├── genai.py
    ├── generate_data.py
    ├── mysql_connection.py
    └── train_model.py