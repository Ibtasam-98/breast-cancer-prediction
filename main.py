import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# Load the dataset (you can replace this with your actual dataset loading code)
@st.cache_data
def load_data():
    # This is just a sample - replace with your actual dataset loading code
    data = pd.read_csv('dataset/breast-cancer.csv')  # Update with your file path
    return data


# Preprocess data and train model
@st.cache_resource
def train_model():
    data = load_data()

    # Convert diagnosis to binary (M = 1, B = 0)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    # Select features - using the most important ones from the dataset
    features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                'smoothness_mean', 'compactness_mean', 'concavity_mean',
                'concave points_mean', 'radius_worst', 'texture_worst']

    X = data[features]
    y = data['diagnosis']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, scaler, accuracy, features


# Create the Streamlit app
def main():
    st.title("Breast Cancer Prediction App")
    st.write("""
    This app predicts whether a breast tumor is **malignant (M)** or **benign (B)** based on cell characteristics.
    Adjust the sliders to input tumor characteristics and click **Predict** to see the result.
    """)

    # Load model and scaler
    model, scaler, accuracy, features = train_model()
    st.sidebar.write(f"Model Accuracy: {accuracy:.2%}")

    # Create sliders for each feature
    st.sidebar.header("Input Tumor Characteristics")

    input_data = {}

    # Radius mean (average radius of tumor cells)
    input_data['radius_mean'] = st.sidebar.slider(
        "Radius Mean (average distance from center to points on the perimeter)",
        6.0, 30.0, 15.0, 0.1)

    # Texture mean (standard deviation of gray-scale values)
    input_data['texture_mean'] = st.sidebar.slider(
        "Texture Mean (standard deviation of gray-scale values)",
        9.0, 40.0, 20.0, 0.1)

    # Perimeter mean
    input_data['perimeter_mean'] = st.sidebar.slider(
        "Perimeter Mean (average size of the core tumor)",
        40.0, 190.0, 100.0, 0.1)

    # Area mean
    input_data['area_mean'] = st.sidebar.slider(
        "Area Mean",
        150.0, 2500.0, 700.0, 1.0)

    # Smoothness mean
    input_data['smoothness_mean'] = st.sidebar.slider(
        "Smoothness Mean (local variation in radius lengths)",
        0.05, 0.20, 0.1, 0.001)

    # Compactness mean
    input_data['compactness_mean'] = st.sidebar.slider(
        "Compactness Mean (perimeter^2 / area - 1.0)",
        0.02, 0.35, 0.1, 0.001)

    # Concavity mean
    input_data['concavity_mean'] = st.sidebar.slider(
        "Concavity Mean (severity of concave portions of the contour)",
        0.0, 0.45, 0.1, 0.001)

    # Concave points mean
    input_data['concave points_mean'] = st.sidebar.slider(
        "Concave Points Mean (number of concave portions of the contour)",
        0.0, 0.2, 0.05, 0.001)

    # Radius worst
    input_data['radius_worst'] = st.sidebar.slider(
        "Radius Worst (largest radius)",
        7.0, 37.0, 20.0, 0.1)

    # Texture worst
    input_data['texture_worst'] = st.sidebar.slider(
        "Texture Worst (worst texture)",
        12.0, 50.0, 25.0, 0.1)

    # Convert input to dataframe
    input_df = pd.DataFrame([input_data])

    # Display user inputs
    st.subheader("User Input Features")
    st.write(input_df)

    # Preprocess input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    if st.button("Predict"):
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        st.subheader("Prediction")
        diagnosis = "Malignant (M)" if prediction[0] == 1 else "Benign (B)"
        st.write(diagnosis)

        st.subheader("Prediction Probability")
        st.write(f"Benign (B): {prediction_proba[0][0]:.2%}")
        st.write(f"Malignant (M): {prediction_proba[0][1]:.2%}")

        # Interpretation
        st.subheader("Interpretation")
        if prediction[0] == 1:
            st.warning(
                "The model predicts this tumor is likely malignant (cancerous). Please consult with a medical professional.")
        else:
            st.success(
                "The model predicts this tumor is likely benign (non-cancerous). However, always consult with a medical professional for proper diagnosis.")


if __name__ == "__main__":
    main()