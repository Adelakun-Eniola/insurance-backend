import pandas as pd
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw" / "dataset_.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "cleaned_dataset_.csv"

df = pd.read_csv(RAW_PATH)
# print(df)
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

df = pd.get_dummies(df, columns=['region'])


def calculate_risk(row):
    score = 0
    if row['bmi'] > 30:
        score += 1
    if row['smoker'] == 1:
        score += 2
    if row['age'] > 50:
        score += 1
    return score


df['risk_score'] = df.apply(calculate_risk, axis=1)

os.mkdir("data/processed")
df.to_csv(PROCESSED_PATH, index= False)

print("pipeline completed!!!")
