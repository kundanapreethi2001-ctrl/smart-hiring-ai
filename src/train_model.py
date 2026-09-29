import pandas as pd
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Load the dataset

df=pd.read_csv("data/candidates.csv")

x=df.drop(columns=["candidate_id","shortlisted"])
y=df["shortlisted"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)

from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train,y_train)
rf_y_pred=rf_model.predict(x_test)
print(rf_y_pred)

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
accuracy_score(y_test,rf_y_pred)
precision_score(y_test,rf_y_pred)
recall_score(y_test,rf_y_pred)
f1_score(y_test,rf_y_pred)

feature_imp=rf_model.feature_importances_
for feature, importance in zip(x.columns, feature_imp):
    print(feature, ":" , importance)

# New candidate
new_candidate = [[
    24,    # age
    2,     # experience
    8.7,   # cgpa
    88,    # coding_score
    90,    # python_score
    78,    # sql_score
    3,     # projects
    1,     # internships
    85     # communication_score
]]

# Prediction
prediction = rf_model.predict(new_candidate)

# Shortlist probability
probability = rf_model.predict_proba(new_candidate)[0][1]

print("Prediction:", prediction)
print("Shortlist Probability:", probability)

import joblib

joblib.dump(rf_model, "models/hiring_model.pkl")

print("Model saved successfully!")