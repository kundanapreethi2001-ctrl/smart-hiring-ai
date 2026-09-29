import numpy as np
import pandas as pd

np.random.seed(42)

n = 1000

data = {
    "candidate_id": [f"C{i:04d}" for i in range(1, n + 1)],
    "age": np.random.randint(21, 30, n),
    "experience": np.random.randint(0, 5, n),
    "cgpa": np.round(np.random.uniform(6.0, 10.0, n), 2),
    "coding_score": np.random.randint(40, 101, n),
    "python_score": np.random.randint(40, 101, n),
    "sql_score": np.random.randint(40, 101, n),
    "projects": np.random.randint(0, 6, n),
    "internships": np.random.randint(0, 3, n),
    "communication_score": np.random.randint(40, 101, n)
}

df = pd.DataFrame(data)

# Create a combined score
df["overall_score"] = (
    df["cgpa"] * 10 * 0.20
    + df["coding_score"] * 0.20
    + df["python_score"] * 0.15
    + df["sql_score"] * 0.10
    + df["communication_score"] * 0.10
    + df["projects"] * 5 * 0.10
    + df["internships"] * 10 * 0.15
)

# Create target variable
df["shortlisted"] = (df["overall_score"] >= 65).astype(int)

# Remove overall_score because it is only used to create the target
df.drop(columns=["overall_score"], inplace=True)

df.to_csv("data/candidates.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("\nShape:", df.shape)
print("\nShortlisted distribution:")
print(df["shortlisted"].value_counts())