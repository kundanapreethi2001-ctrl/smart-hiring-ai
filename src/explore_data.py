import os
import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Load the dataset
df = pd.read_csv("data/candidates.csv")

# 1. First 5 rows
print("FIRST 5 ROWS")
print(df.head())

# 2. Number of rows and columns
print("\nSHAPE")
print(df.shape)

# 3. Column names
print("\nCOLUMNS")
print(df.columns)

# 4. Data types
print("\nDATA TYPES")
print(df.dtypes)

# 5. Basic statistics
print("\nSTATISTICS")
print(df.describe())

# 6. Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# 7. Shortlisted distribution
print("\nSHORTLISTED DISTRIBUTION")
print(df["shortlisted"].value_counts())

print(df["coding_score"].max())
print(df["python_score"].max())
print(df["sql_score"].max())

print((df["cgpa"]>8.5).sum())
print(df["coding_score"].mean())
print(df[df["shortlisted"]==1]["coding_score"].mean())
print(df[df["shortlisted"]==0]["coding_score"].mean())
print((df["python_score"]>80).sum())
print(df[df["shortlisted"]==1]["experience"].mean())
print((df[df["shortlisted"]==1]["projects"]>2).sum())
print(df.groupby("shortlisted")["sql_score"].mean())
df.groupby("shortlisted")["experience"].mean()
df.groupby("shortlisted")["shortlisted"].count()
df["coding_score"].mean()
df["python_score"].mean()
df["sql_score"].mean()
df["experience"].mean()
df["cgpa"].mean()





