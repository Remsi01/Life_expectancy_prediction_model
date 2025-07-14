import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Function to load the model components
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_model(model_path='life_expectancy_model.pkl'):
    """
    Loads the saved model, scaler, and features.
    """
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at '{model_path}'. Please ensure it's in the same directory.")
        return None, None, None
    try:
        model_data = joblib.load(model_path)
        return model_data['model'], model_data['scaler'], model_data['features']
    except Exception as e:
        st.error(f"Error loading model components: {e}. Model file might be corrupted.")
        return None, None, None

# Function to make predictions
def predict_life_expectancy(female_le, male_le, model, scaler, features):
    """
    Predicts the total life expectancy (both sexes) using the loaded model components.
    """
    # Create a DataFrame for the new input, ensuring column order matches training
    input_data = pd.DataFrame([[female_le, male_le]], columns=features)

    # Scale the new input data using the *same scaler* that was fitted on training data
    input_data_scaled = scaler.transform(input_data)

    # Make a prediction
    prediction = model.predict(input_data_scaled)
    return prediction[0]

# --- Streamlit App Layout ---
st.set_page_config(page_title="Life Expectancy Predictor", layout="centered")

st.title("🌍 Life Expectancy Predictor")
# Load model components once
model, scaler, features = load_model()

if model is None or scaler is None or features is None:
    st.warning("Model could not be loaded. Please ensure 'life_expectancy_model.pkl' is in the correct directory.")
else:
    st.subheader("Enter Life Expectancy Data:")

    # Input fields for female and male life expectancy
    # Use st.number_input for numerical inputs
    female_le_input = st.number_input(
        "Female Life Expectancy (years)",
        min_value=30.0, max_value=95.0, value=78.0, step=0.1,
        help="Enter an average life expectancy for females (e.g., 78.5)."
    )
    male_le_input = st.number_input(
        "Male Life Expectancy (years)",
        min_value=30.0, max_value=95.0, value=74.0, step=0.1,
        help="Enter an average life expectancy for males (e.g., 74.0)."
    )

    # Prediction button
    if st.button("Predict Total Life Expectancy"):
        # Basic validation for input range, matching the filtering applied during training
        min_bound = 30
        max_bound = 95

        if not (min_bound <= female_le_input <= max_bound and min_bound <= male_le_input <= max_bound):
            st.error(f"Please enter values within the realistic range ({min_bound}-{max_bound} years) for better prediction reliability.")
        else:
            # Make prediction
            predicted_value = predict_life_expectancy(female_le_input, male_le_input, model, scaler, features)

            st.success(f"**Predicted Total Life Expectancy (both sexes): {predicted_value:.2f} years**")

    st.markdown("---")
