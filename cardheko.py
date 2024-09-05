import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Load the trained models
model_files = {
    'Linear Regression': 'C:/Users/sonur/OneDrive/Desktop/Car_Dheko/Models/trained_model.pkl',
    'Ridge Regression': 'C:/Users/sonur/OneDrive/Desktop/Car_Dheko/Models/Ridge_Regression_model.pkl',
    'Lasso Regression': 'C:/Users/sonur/OneDrive/Desktop/Car_Dheko/Models/Lasso_Regression_model.pkl',
    'Random Forest': 'C:/Users/sonur/OneDrive/Desktop/Car_Dheko/Models/Random_Forest_model.pkl',
    'Gradient Boosting': 'C:/Users/sonur/OneDrive/Desktop/Car_Dheko/Models/Gradient_Boosting_model.pkl'
}

models = {name: joblib.load(path) for name, path in model_files.items()}

# Load preprocessing tools
scaler = joblib.load('C:/Users/sonur/OneDrive/Desktop/Car_Dheko/Models/scaler.pkl')
imputer = joblib.load('C:/Users/sonur/OneDrive/Desktop/Car_Dheko/Models/imputer.pkl')
poly = joblib.load('C:/Users/sonur/OneDrive/Desktop/Car_Dheko/Models/poly.pkl')
pca = joblib.load('C:/Users/sonur/OneDrive/Desktop/Car_Dheko/Models/pca.pkl')

# Define Streamlit UI
st.title('Car Price Prediction App')

# Input fields for user
st.sidebar.header('Car Features Input')

city = st.sidebar.selectbox('City', options=['City1', 'City2', 'City3'])
fuel_type = st.sidebar.selectbox('Fuel Type', options=['Petrol', 'Diesel', 'Electric'])
body_type = st.sidebar.selectbox('Body Type', options=['Sedan', 'SUV', 'Hatchback'])
kilometers_driven = st.sidebar.number_input('Kilometers Driven', min_value=0)
transmission = st.sidebar.selectbox('Transmission', options=['Manual', 'Automatic'])
owner = st.sidebar.selectbox('Owner', options=['1st Owner', '2nd Owner', '3rd Owner'])
oem = st.sidebar.selectbox('OEM', options=['Brand1', 'Brand2', 'Brand3'])
model = st.sidebar.selectbox('Model', options=['Model1', 'Model2', 'Model3'])
year = st.sidebar.number_input('Year of Manufacture', min_value=1900, max_value=2024)
variant = st.sidebar.text_input('Variant')

# Prepare input data for prediction
input_data = pd.DataFrame({
    'City': [city],
    'fuel_type': [fuel_type],
    'body_type': [body_type],
    'kilometers_driven': [kilometers_driven],
    'transmission': [transmission],
    'owner': [owner],
    'oem': [oem],
    'model': [model],
    'year': [year],
    'variant': [variant]
})

# Convert categorical variables to dummies
input_data = pd.get_dummies(input_data, drop_first=True)

# Impute missing values
input_data_imputed = imputer.transform(input_data)

# Add polynomial features if needed
input_data_poly = poly.transform(input_data_imputed)

# Standardize features
input_data_scaled = scaler.transform(input_data_poly)

# Dimensionality Reduction with PCA
input_data_pca = pca.transform(input_data_scaled)

# Predict prices with all models
predictions = {name: model.predict(input_data_pca)[0] for name, model in models.items()}

# Display predictions
st.header('Predicted Car Prices')
for name, prediction in predictions.items():
    st.write(f"{name}: ₹ {prediction:,.2f}")

# Optionally, allow users to select a model
selected_model = st.sidebar.selectbox('Select Model', options=list(models.keys()))
if st.button('Predict Price'):
    if selected_model:
        model = models[selected_model]
        price = model.predict(input_data_pca)[0]
        st.write(f"The predicted price using {selected_model} is ₹ {price:,.2f}")
    else:
        st.write("Please select a model to get the prediction.")
