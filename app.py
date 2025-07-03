import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

# Configuration
MODELS_DIR = "saved_models/all_features"
HIGHLY_CORRELATED_FEATURES = [
    'concave points_worst', 'perimeter_worst', 'concave points_mean',
    'radius_worst', 'perimeter_mean', 'area_worst', 'radius_mean', 'area_mean'
]


# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('dataset/breast-cancer.csv')
    data = data.drop(['id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


# Load trained models and scaler
@st.cache_resource
def load_models_and_scaler():
    model_files = {
        'Gradient Boosting': 'gradient_boosting.pkl',
        'KNN': 'knn.pkl',
        'Random Forest': 'random_forest.pkl',
        'SGD Classifier': 'sgd_classifier.pkl',
        'Stacking': 'stacking.pkl',
        'SVM Linear': 'svm_linear.pkl',
        'SVM RBF': 'svm_rbf.pkl'
    }

    models = {}

    # Load models
    for name, file in model_files.items():
        path = os.path.join(MODELS_DIR, file)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)

    # Load scaler
    scaler = None
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

    return models, scaler


def create_feature_sliders(data):
    input_data = {}
    stats = data.describe()

    st.sidebar.subheader("Highly Correlated Features")
    st.sidebar.caption("These features have the strongest relationship with diagnosis")

    # Create sliders with more extreme default values to see impact
    for feature in HIGHLY_CORRELATED_FEATURES:
        if feature in data.columns:
            min_val = stats[feature]['min']
            max_val = stats[feature]['max']
            mean_val = stats[feature]['mean']

            # Set default to 25th percentile (more likely benign) or 75th percentile (more likely malignant)
            default_val = mean_val  # Start with mean as default

            input_data[feature] = st.sidebar.slider(
                label=f"⭐ {feature.replace('_', ' ').title()} ⭐",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val),
                step=float((max_val - min_val) / 100),
                key=f"hc_{feature}"
            )

    return input_data


def main():
    st.set_page_config(layout="wide")
    st.title("Breast Cancer Prediction App")
    st.write("""
    This app predicts whether a breast tumor is **malignant (M)** or **benign (B)** using multiple machine learning algorithms.
    Adjust the sliders to see how different feature values affect the prediction.
    """)

    # Load data and models
    data = load_data()
    models, scaler = load_models_and_scaler()

    if not models:
        st.error("No trained models found. Please train models first.")
        return
    if scaler is None:
        st.error(
            "Scaler not found. Please ensure you have a scaler.pkl file in your saved_models/all_features directory.")
        return

    # Create input sliders in sidebar
    input_data = create_feature_sliders(data)

    # Main content area
    col1, col2 = st.columns([3, 2])

    with col1:
        st.header("Model Information")
        st.write("The following models are available for prediction:")
        model_df = pd.DataFrame({
            "Model": list(models.keys()),
            "Type": ["Ensemble", "Neighbors", "Ensemble", "Linear",
                     "Ensemble", "SVM", "SVM"]
        })
        st.dataframe(model_df, use_container_width=True, hide_index=True)

    with col2:
        st.header("Key Features Info")
        st.write("These features are most important for diagnosis:")
        hc_features_df = pd.DataFrame({
            "Feature": HIGHLY_CORRELATED_FEATURES,
            "Importance": ["High"] * len(HIGHLY_CORRELATED_FEATURES)
        })
        st.dataframe(hc_features_df, use_container_width=True, hide_index=True)

    # Display user inputs
    st.subheader("Your Input Values")
    st.dataframe(pd.DataFrame([input_data]).style.format("{:.2f}"), use_container_width=True)

    # Make predictions
    if st.button("Predict Diagnosis", type="primary", use_container_width=True):
        # Prepare input dataframe with only the features we have sliders for
        input_df = pd.DataFrame([input_data])

        try:
            # Get the intersection of features we have and features the scaler expects
            available_features = [f for f in input_df.columns if f in scaler.feature_names_in_]
            input_df = input_df[available_features]

            # Add any missing features with their mean values
            stats = data.describe()
            missing_features = [f for f in scaler.feature_names_in_ if f not in input_df.columns]
            for feature in missing_features:
                input_df[feature] = stats[feature]['mean']

            # Reorder columns to match scaler's expected order
            input_df = input_df[scaler.feature_names_in_]
            input_scaled = scaler.transform(input_df)

            # Debug: Show scaled values being used
            with st.expander("Debug: Scaled Feature Values"):
                st.write("These are the scaled values being sent to the models:")
                scaled_df = pd.DataFrame(input_scaled, columns=scaler.feature_names_in_)
                st.dataframe(scaled_df.style.format("{:.2f}"))

                # Show feature means for comparison
                st.write("Feature means from training data:")
                means = pd.DataFrame([stats.loc['mean']]).drop('diagnosis', axis=1)
                st.dataframe(means[scaler.feature_names_in_].style.format("{:.2f}"))

        except Exception as e:
            st.error(f"Feature processing error: {str(e)}")
            st.error(f"Expected features: {scaler.feature_names_in_}")
            st.error(f"Provided features: {input_df.columns.tolist()}")
            return

        st.subheader("Model Predictions")

        # Create prediction results
        predictions = []
        for model_name, model in models.items():
            try:
                pred = model.predict(input_scaled)[0]
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_scaled)[0]
                    confidence = max(proba)
                    benign_prob = proba[0]
                    malignant_prob = proba[1]
                else:
                    confidence = 1.0
                    benign_prob = 0.0 if pred == 1 else 1.0
                    malignant_prob = 1.0 if pred == 1 else 0.0

                predictions.append({
                    'Model': model_name,
                    'Prediction': 'Malignant' if pred == 1 else 'Benign',
                    'Confidence': f"{confidence * 100:.1f}%",
                    'Benign Probability': f"{benign_prob * 100:.1f}%",
                    'Malignant Probability': f"{malignant_prob * 100:.1f}%"
                })
            except Exception as e:
                st.warning(f"Error with {model_name}: {str(e)}")
                continue

        # Display predictions
        if predictions:
            pred_df = pd.DataFrame(predictions)
            st.dataframe(pred_df, use_container_width=True)

            # Calculate consensus
            final_pred = np.mean([1 if p['Prediction'] == 'Malignant' else 0 for p in predictions])
            confidence = np.mean([float(p['Confidence'].strip('%')) / 100 for p in predictions])

            st.subheader("Final Consensus Prediction")
            if final_pred > 0.5:
                st.error(f"## Consensus: Malignant (M) with {confidence * 100:.1f}% confidence")
                st.warning("""
                **Clinical Recommendation:** 
                The models suggest a high probability of malignancy. 
                Please consult with an oncologist immediately for further evaluation.
                """)
            else:
                st.success(f"## Consensus: Benign (B) with {confidence * 100:.1f}% confidence")
                st.info("""
                **Clinical Recommendation:** 
                The models suggest a high probability of benign tumor. 
                Regular monitoring is still recommended.
                """)

            # Show feature importance for tree-based models
            st.subheader("Feature Importance (Random Forest)")
            if 'Random Forest' in models:
                try:
                    importances = models['Random Forest'].feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': input_df.columns,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)

                    st.bar_chart(importance_df.set_index('Feature'))

                    # Show which features were adjusted by the user
                    importance_df['User Adjusted'] = importance_df['Feature'].isin(HIGHLY_CORRELATED_FEATURES)
                    st.write("Features highlighted in yellow were adjusted in the sidebar:")
                    st.dataframe(importance_df.style.apply(
                        lambda x: ['background: yellow' if x['User Adjusted'] else '' for i in x],
                        axis=1
                    ))
                except Exception as e:
                    st.warning(f"Could not display feature importance: {str(e)}")
        else:
            st.error("No models produced valid predictions.")


if __name__ == "__main__":
    main()