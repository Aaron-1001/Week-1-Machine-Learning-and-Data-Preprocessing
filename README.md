# Week 1 — Python for Machine Learning & Data Preprocessing

## Project
This project demonstrates Python, NumPy, Pandas, exploratory data analysis, missing-value handling, categorical encoding, and numeric feature scaling on a small **synthetic student-performance dataset**.

## Files
- `student_performance_sample.csv` — input sample dataset (20 rows).
- `project.py` — complete preprocessing and EDA script.
- `cleaned_student_data.csv` — generated when the script runs.
- `final_score_distribution.png` — generated histogram.

## Setup and run
```bash
python -m pip install -r requirements.txt
python project.py
```

## Processing steps
1. Read the CSV with Pandas and inspect shape, data types, and descriptive statistics.
2. Check missing values and duplicate records.
3. Fill missing numeric values with each column's median.
4. Fill missing categorical values with each column's mode.
5. Remove duplicate rows if present.
6. One-hot encode `Department` and `InternetAccess`.
7. Scale numeric columns to the 0–1 range using `MinMaxScaler`.
8. Plot the final-score distribution and save the processed CSV.

