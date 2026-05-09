import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from shap import plots

st.set_page_config(page_title="🏡 House Price Estimator", layout="wide")
st.title("🏡 Ames House Price Prediction Dashboard")
st.caption("Enter property details → Get instant AI valuation + feature impact")

# Load model & preprocessor
@st.cache_resource
def load_assets():
    model = joblib.load("house_price_model.pkl")
    preprocessor = joblib.load("house_price_preprocessor.pkl")
    feat_names = joblib.load("feature_names.pkl")
    nb_map = joblib.load("nb_mapping.pkl")
    return model, preprocessor, feat_names, nb_map

model, preprocessor, feat_names, nb_map = load_assets()

# Sidebar Inputs
st.sidebar.header("🔑 Property Details")
overall_qual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 7)
gr_liv_area = st.sidebar.number_input("Above Ground Living Area (sq ft)", 300, 5000, 1500)
neighborhood = st.sidebar.selectbox("Neighborhood", nb_map.index.tolist(), 5)
total_bsmt_sf = st.sidebar.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
garage_cars = st.sidebar.slider("Garage Cars", 0, 4, 2)
year_built = st.sidebar.number_input("Year Built", 1872, 2010, 1990)
full_bath = st.sidebar.slider("Full Bathrooms", 0, 4, 2)

# Build input DataFrame (match training schema)
input_data = pd.DataFrame([{
    'OverallQual': overall_qual,
    'GrLivArea': gr_liv_area,
    'Neighborhood': neighborhood,
    'TotalBsmtSF': total_bsmt_sf,
    'GarageCars': garage_cars,
    'YearBuilt': year_built,
    'FullBath': full_bath,
    # Fill remaining with median defaults (simplified)
    **{col: 0 for col in ['MasVnrArea', 'BsmtFinSF1', 'GarageArea']}
}])

# Add engineered features
input_data['TotalSF'] = input_data['TotalBsmtSF'] + 1500 + 0  # Approx
input_data['HouseAge'] = 2024 - input_data['YearBuilt']
input_data['QualArea'] = input_data['OverallQual'] * input_data['GrLivArea']
input_data['Neighborhood_Encoded'] = input_data['Neighborhood'].map(nb_map).fillna(nb_map.median())

# Predict
if st.button("🔮 Predict Price"):
    with st.spinner("Running model..."):
        # Reorder columns to match preprocessor
        input_p = preprocessor.transform(input_data[preprocessor.feature_names_in_])
        pred_log = model.predict(input_p)[0]
        pred_dollars = np.expm1(pred_log)
        
        st.metric("💰 Estimated Sale Price", f"${pred_dollars:,.0f}")
        
        # SHAP for this input
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(input_p)
        
        st.subheader("📊 What drove this prediction?")
        shap_html = plots.force(explainer.expected_value, sv[0], input_p[0], 
                                feature_names=feat_names, matplotlib=False)
        st.components.v1.html(shap_html.html(), height=300, scrolling=False)