# save_models.py
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier

# Re-run your pipeline to get trained objects
# (Or load from run_fraud.py if you saved them there)

# Example: If you already trained these in run_fraud.py:
# model = xgb, iso_forest = iso, scaler = scaler

# For demo, we'll retrain minimal versions:
df = pd.read_csv("creditcard.csv")
df['Amount_scaled'] = RobustScaler().fit_transform(df[['Amount']])
pca_cols = [f'V{i}' for i in range(1, 29)]

# Fit IsolationForest
iso = IsolationForest(n_estimators=50, contamination=0.01, random_state=42, n_jobs=-1)
iso.fit(df[pca_cols + ['Amount_scaled']])

# Fit XGBoost (minimal)
from sklearn.model_selection import train_test_split
X = df[pca_cols + ['Amount_scaled']]
y = df['Class']
X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
xgb = XGBClassifier(n_estimators=50, max_depth=3, scale_pos_weight=(y_tr==0).sum()/(y_tr==1).sum(), random_state=42, n_jobs=-1)
xgb.fit(X_tr, y_tr)

# Save scaler separately
scaler = RobustScaler()
scaler.fit(df[['Amount']])

# Save all
import os
os.makedirs("models", exist_ok=True)
joblib.dump(xgb, "models/xgb_fraud.pkl")
joblib.dump(iso, "models/iso_forest.pkl")
joblib.dump(scaler, "models/robust_scaler.pkl")
joblib.dump({'threshold': 0.32, 'features': pca_cols + ['Amount_scaled']}, "models/config.pkl")
print("✅ Models saved to /models/")