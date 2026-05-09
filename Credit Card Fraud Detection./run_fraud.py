# run_fraud.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🚀 Starting Credit Card Fraud Detection Pipeline...")
    
    # 1. Load Data
    data_path = "creditcard.csv"
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found. Place it in D:\\car\\4\\")
        return
        
    print("📥 Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df):,} transactions. Fraud rate: {df['Class'].mean():.4%}")
    
    # 2. Feature Engineering
    print("🔧 Engineering features...")
    df['Time_hours'] = df['Time'] / 3600
    df['Amount_log'] = np.log1p(df['Amount'])
    
    scaler = RobustScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    
    pca_cols = [f'V{i}' for i in range(1, 29)]
    iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
    iso.fit(df[pca_cols + ['Amount_scaled']])  # ✅ ADD THIS LINE
    df['anomaly_score'] = iso.decision_function(df[pca_cols + ['Amount_scaled']])
    # 3. Time-Aware Split
    print("⏱️ Splitting chronologically...")
    df_sorted = df.sort_values('Time').reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)
    
    features = [c for c in df.columns if c not in ['Time', 'Amount', 'Class']]
    X_train = df_sorted[features].iloc[:split_idx]
    X_val = df_sorted[features].iloc[split_idx:]
    y_train = df_sorted['Class'].iloc[:split_idx]
    y_val = df_sorted['Class'].iloc[split_idx:]
    
    print(f"✅ Train: {len(X_train):,} | Val: {len(X_val):,}")
    
    # 4. Train XGBoost (Cost-Sensitive)
    print("🧠 Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric='logloss',
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    val_proba = model.predict_proba(X_val)[:, 1]
    print(f"✅ Val ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}")
    
    # 5. Financial Cost Threshold Optimization
    print("💰 Optimizing threshold for minimum $ loss...")
    amounts_val = df_sorted['Amount'].iloc[split_idx:].values
    best_cost, optimal_thresh = float('inf'), 0.5
    
    for t in np.linspace(0.1, 0.8, 80):
        preds = (val_proba >= t).astype(int)
        fp = ((preds == 1) & (y_val == 0)).sum()
        fn_amt = amounts_val[(preds == 0) & (y_val == 1)]
        cost = (fp * 2.0) + (fn_amt * 0.8).sum()  # $2 review, 80% unrecovered
        if cost < best_cost:
            best_cost, optimal_thresh = cost, t
            
    val_preds = (val_proba >= optimal_thresh).astype(int)
    print(f"✅ Optimal Threshold: {optimal_thresh:.3f} | Min Loss: ${best_cost:,.2f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_val, val_preds, target_names=['Legit', 'Fraud']))
    
    # 6. Save Artifacts
    print("💾 Saving models...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_fraud.pkl")
    joblib.dump(iso, "models/iso_forest.pkl")
    joblib.dump(scaler, "models/robust_scaler.pkl")
    joblib.dump({'threshold': optimal_thresh, 'features': features}, "models/config.pkl")
    print("🎉 Pipeline complete! Models saved in /models/")

if __name__ == "__main__":
    main()