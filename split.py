import os
import pandas as pd
from sklearn.model_selection import train_test_split

os.makedirs("data/processed", exist_ok=True)
df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")
X, y = df.drop(columns="target"), df["target"]
a, b, c, d = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
a.to_csv("data/processed/X_train.csv", index=False)
b.to_csv("data/processed/X_test.csv", index=False)
c.to_csv("data/processed/y_train.csv", index=False)
d.to_csv("data/processed/y_test.csv", index=False)