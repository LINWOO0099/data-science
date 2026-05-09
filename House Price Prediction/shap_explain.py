import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1️⃣ Load saved artifacts
model = joblib.load("house_price_model.pkl")
preprocessor = joblib.load("house_price_preprocessor.pkl")
feat_names = joblib.load("feature_names.pkl")
nb_map = joblib.load("nb_mapping.pkl")

# 2️⃣ Load raw training data
train = pd.read_csv("train.csv")
train['SalePrice_log'] = np.log1p(train['SalePrice'])

# 🔧 REAPPLY CLEANING (Must match training pipeline exactly)
# LotFrontage → neighborhood median
train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)

# "None" structural features
none_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'GarageType', 'PoolQC', 'Fence', 'MiscFeature']
for col in none_cols: train[col] = train[col].fillna('None')

# Zero-area features
zero_cols = ['MasVnrArea', 'BsmtFinSF1', 'GarageArea', 'TotalBsmtSF']
for col in zero_cols: train[col] = train[col].fillna(0)

# Remaining fills
for col in train.select_dtypes('object').columns: train[col] = train[col].fillna(train[col].mode()[0])
for col in train.select_dtypes('number').columns: train[col] = train[col].fillna(train[col].median())

# 🔧 REAPPLY FEATURE ENGINEERING
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
train['TotalBath'] = train['FullBath'] + train['HalfBath'] + train['BsmtFullBath'] + train['BsmtHalfBath']
train['HouseAge'] = 2024 - train['YearBuilt']
train['IsRemodeled'] = (train['YearRemodAdd'] > train['YearBuilt']).astype(int)
train['QualArea'] = train['OverallQual'] * train['GrLivArea']
for col in ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'TotalBsmtSF', 'GarageArea']:
    train[f'{col}_log'] = np.log1p(train[col])
train['Neighborhood_Encoded'] = train['Neighborhood'].map(nb_map).fillna(nb_map.median())

# 3️⃣ Split & Transform Validation Set (THIS CREATES X_val_p)
drop_cols = ['Id', 'SalePrice', 'SalePrice_log']
X = train.drop(columns=drop_cols)
y = train['SalePrice_log']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_val_p = preprocessor.transform(X_val)  # ✅ This is what was missing!

# 4️⃣ SHAP EXPLANATIONS
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val_p[:100])  # Use first 100 validation houses

# 🌍 Global Feature Importance
shap.summary_plot(shap_values, X_val_p[:100], feature_names=feat_names, show=False)
plt.title("Global Feature Importance (SHAP)")
plt.tight_layout()
plt.show()

# 🏠 Individual Prediction (House #0 in validation set)
shap.force_plot(explainer.expected_value, shap_values[0], 
                X_val_p[0], feature_names=feat_names, matplotlib=True)
plt.title("SHAP Explanation: Validation House #1")
plt.tight_layout()
plt.show()