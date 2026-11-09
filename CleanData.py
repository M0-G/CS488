import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")

# 1. Remove duplicate rows (before splitting, so copies can't leak into test)
df = df.drop_duplicates().reset_index(drop=True)

# 2. Split first (stratified, seed 42)
X, y = df.drop(columns="target"), df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_test = X_train.copy(), X_test.copy()

# 3. Treat impossible zeros as missing, fill with the TRAIN median
for col in ["cholesterol", "resting bp s"]:
    X_train[col] = X_train[col].replace(0, np.nan)
    X_test[col] = X_test[col].replace(0, np.nan)
    median = X_train[col].median()          # learned from train only
    X_train[col] = X_train[col].fillna(median)
    X_test[col] = X_test[col].fillna(median)

# 4. Save
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Rows after removing duplicates:", len(df))
print("Train/test sizes:", len(X_train), len(X_test))