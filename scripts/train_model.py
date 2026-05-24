"""
Train the visit-prioritization model.

Input  : data/training.csv
Output : model/visit_model.joblib
         model/feature_importance.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "training.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
print(f"Loaded {len(df):,} rows, positive rate = {df['label_order_next_7d'].mean():.2%}")

# Feature engineering -------------------------------------------------------
df["tier_score"] = df["tier"].map({"A": 3, "B": 2, "C": 1})
df["crop_chilli"] = (df["primary_crop"] == "chilli").astype(int)
df["crop_maize"]  = (df["primary_crop"] == "maize").astype(int)
df["crop_cotton"] = (df["primary_crop"] == "cotton").astype(int)

# NDVI feature: merge latest weekly NDVI per district as of each training date.
ndvi = pd.read_csv(ROOT / "data" / "ndvi.csv", parse_dates=["week_start"])
ndvi = ndvi.sort_values("week_start")
df = df.copy()
df["date_dt"] = pd.to_datetime(df["date"])
df_sorted = df.sort_values("date_dt")
# Merge_asof gets the NDVI for the most recent week_start <= date for each district.
df = pd.merge_asof(df_sorted, ndvi, left_on="date_dt", right_on="week_start",
                   by="district", direction="backward")
df["ndvi"] = df["ndvi"].fillna(0.3)  # fallback if no NDVI record found
df.drop(columns=["date_dt", "week_start"], inplace=True)

FEATURES = [
    "avg_monthly_sales_inr",
    "credit_score",
    "days_since_visit",
    "recent_rain_7d_mm",
    "recent_humidity_pct",
    "pest_pressure_14d",
    "ndvi",
    "tier_score",
    "crop_chilli",
    "crop_maize",
    "crop_cotton",
]
TARGET = "label_order_next_7d"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.08,
    random_state=42,
)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)
print(f"\nTest ROC-AUC = {auc:.3f}")
print(classification_report(y_test, (proba > 0.5).astype(int), digits=3))

# Feature importance for explainability -------------------------------------
importance = sorted(
    zip(FEATURES, model.feature_importances_),
    key=lambda x: -x[1],
)
print("\nFeature importance:")
for name, imp in importance:
    print(f"  {name:30s} {imp:.3f}")

# Persist -------------------------------------------------------------------
joblib.dump({"model": model, "features": FEATURES, "auc": auc}, MODEL_DIR / "visit_model.joblib")
with (MODEL_DIR / "feature_importance.json").open("w") as f:
    json.dump([{"feature": n, "importance": float(i)} for n, i in importance], f, indent=2)

print(f"\nSaved -> {MODEL_DIR/'visit_model.joblib'}")
