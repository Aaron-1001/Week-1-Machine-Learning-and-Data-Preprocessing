"""
Week 1 Task: Python for Machine Learning & Data Preprocessing
Run: python preprocess.py
Requirements: pandas, numpy, matplotlib, scikit-learn
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "student_performance_sample.csv"
OUTPUT_FILE = BASE_DIR / "cleaned_student_data.csv"

# 1. Load data with Pandas
data = pd.read_csv(INPUT_FILE)
print("First five rows:\n", data.head())
print("\nDataset shape:", data.shape)
print("\nData types and non-null counts:")
data.info()
print("\nNumerical summary:\n", data.describe())

# 2. Explore missing values and duplicates
print("\nMissing values before cleaning:\n", data.isnull().sum())
print("\nDuplicate rows:", data.duplicated().sum())

# 3. Handle missing values
numeric_cols = ["Age", "StudyHours", "Attendance", "FinalScore"]
categorical_cols = ["Department", "InternetAccess"]

for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# Remove duplicate records, if any
data = data.drop_duplicates()

# 4. Encode categorical variables using one-hot encoding
data_encoded = pd.get_dummies(
    data, columns=categorical_cols, dtype=int
)

# 5. Normalize numeric features to the 0-1 range
scaler = MinMaxScaler()
data_encoded[numeric_cols] = scaler.fit_transform(data_encoded[numeric_cols])

# 6. Exploratory data analysis (EDA)
print("\nMissing values after cleaning:\n", data_encoded.isnull().sum())
print("\nCleaned and encoded data:\n", data_encoded.head())
print("\nCorrelation matrix:\n", data[numeric_cols].corr())

plt.figure(figsize=(7, 4))
data["FinalScore"].hist(bins=8, edgecolor="black")
plt.title("Distribution of Final Scores")
plt.xlabel("Final score")
plt.ylabel("Number of students")
plt.tight_layout()
plt.savefig(BASE_DIR / "final_score_distribution.png")
plt.close()

# 7. Save the processed dataset
data_encoded.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved processed dataset to: {OUTPUT_FILE}")
