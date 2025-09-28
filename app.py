import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import StackingClassifier
from matplotlib.colors import LinearSegmentedColormap 

# Set page config
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon=":hospital:",
    layout="wide"
)

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_SAVE_PATH = "saved_models"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Highly correlated features (from your analysis)
HIGHLY_CORR_FEATURES = [
    'concave points_worst',
    'perimeter_worst',
    'concave points_mean',
    'radius_worst',
    'perimeter_mean',
    'area_worst',
    'radius_mean',
    'area_mean'
]


# Load data
@st.cache_data
def load_data():
    """Load and preprocess the breast cancer dataset"""
    try:
        data = pd.read_csv('dataset/breast-cancer.csv')
        data = data.drop(['id'], axis=1)
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

        # Select only highly correlated features
        features = [f for f in HIGHLY_CORR_FEATURES if f in data.columns]
        # CORRECTED LINE: Ensure data assignment is correct
        data = data[features + ['diagnosis']]

        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


# Model definitions
def get_models():
    """Return dictionary of models and their parameter grids"""
    models = {
        'SVM Linear': {
            'model': SVC(random_state=RANDOM_STATE, probability=True),
            'params': {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
        },
        'SVM RBF': {
            'model': SVC(random_state=RANDOM_STATE, probability=True),
            'params': {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1]},
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {'n_neighbors': range(3, 15), 'weights': ['uniform', 'distance']},
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=RANDOM_STATE),
            'params': {'C': [0.1, 1, 10], 'penalty': ['l2']},
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 0.5],
                'max_depth': [3, 5]
            },
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            },
        },
    }
    return models


# Train and evaluate model
def train_model(model, params, X_train, y_train, X_test, y_test):
    """Train and evaluate a single model"""
    start_time = time.time()

    grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)

    metrics = {
        'train_accuracy': accuracy_score(y_train, train_pred),
        'test_accuracy': accuracy_score(y_test, test_pred),
        'training_time': time.time() - start_time,
        'best_params': grid.best_params_,
        'model': best_model,
        'classification_report': classification_report(y_test, test_pred, target_names=['Benign', 'Malignant']),
        'confusion_matrix': confusion_matrix(y_test, test_pred)
    }
    return metrics


# Plot functions
def plot_confusion_matrix(cm, model_name):
    """Plot a single confusion matrix"""
    fig, ax = plt.subplots(figsize=(6, 4))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'],
                cbar=False, ax=ax)

    # CORRECTED LINE: Removed .data.data.frame()
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j + 0.5, i + 0.25, f"{cm[i, j]}",
                    ha='center', va='center', color='black')

    ax.set_title(f'{model_name}', pad=20)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    return fig


def plot_roc_curve(models, X_test, y_test):
    """Plot ROC curves for all models"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--')

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc="lower right")
    return fig


# Main function
def main():
    st.title("Breast Cancer Classification")
    st.markdown("""
    This app uses machine learning to classify breast cancer tumors as benign or malignant
    using only the most highly correlated features.
    """)

    # Load data
    data = load_data()
    if data is None:
        return

    # Split data
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    if st.button("Train All Models"):
        st.info("Training models with hyperparameter tuning... This may take a few minutes.")

        models = get_models()
        results = []
        trained_models = {}

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, (name, model_def) in enumerate(models.items()):
            status_text.text(f"Training {name}...")

            metrics = train_model(
                model_def['model'],
                model_def['params'],
                X_train_scaled,
                y_train,
                X_test_scaled,
                y_test
            )

            results.append({
                'Model': name,
                'Train Accuracy': metrics['train_accuracy'],
                'Test Accuracy': metrics['test_accuracy'],
                'Training Time (s)': metrics['training_time']
            })

            # CORRECTED LINE: Removed .data.data.frame()
            trained_models[name] = metrics['model']

            # Save model
            with open(os.path.join(MODEL_SAVE_PATH, f"{name.lower().replace(' ', '_')}.pkl"), 'wb') as f:
                pickle.dump(metrics['model'], f)

            progress_bar.progress((i + 1) / len(models))

        # Save scaler
        with open(os.path.join(MODEL_SAVE_PATH, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

        status_text.text("Training complete!")

        # Show results
        st.subheader("Model Performance")
        # CORRECTED LINE: Removed .data.data.frame()
        results_df = pd.DataFrame(results).sort_values('Test Accuracy', ascending=False)
        st.dataframe(results_df.style.format({
            'Train Accuracy': '{:.2%}',
            'Test Accuracy': '{:.2%}',
            'Training Time (s)': '{:.2f}'
        }))

        # Show best model
        # CORRECTED LINE: Removed .ilocdata.data.frame()
        best_model_name = results_df.iloc[0]['Model']
        st.success(f"Best model: {best_model_name} with test accuracy of {results_df.iloc[0]['Test Accuracy']:.2%}")

        # Show classification report for best model
        st.subheader(f"Classification Report for {best_model_name}")
        # CORRECTED LINE: Removed .data.data.frame()
        best_model = trained_models[best_model_name]
        y_pred = best_model.predict(X_test_scaled)
        st.text(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

        # Confusion matrices in a horizontal layout
        st.subheader("Model Confusion Matrices")
        cols = st.columns(len(trained_models))
        for (name, model), col in zip(trained_models.items(), cols):
            with col:
                y_pred = model.predict(X_test_scaled)
                cm = confusion_matrix(y_test, y_pred)
                fig = plot_confusion_matrix(cm, name)
                st.pyplot(fig)

        # ROC curve
        st.subheader("ROC Curves")
        roc_fig = plot_roc_curve(trained_models, X_test_scaled, y_test)
        st.pyplot(roc_fig)

        # Feature correlations with dark blue/light blue colors (highlighting strong negative)
        st.subheader("Feature Correlations")
        corr_fig, ax = plt.subplots(figsize=(10, 8))

        # Define a custom diverging colormap: very dark blue -> white -> light blue
        # #000080 is a dark navy blue, lightblue is a common light blue, white for zero, lightskyblue for positive
        colors = ["#000080", "lightblue", "white", "lightskyblue"]
        cmap = LinearSegmentedColormap.from_list("dark_blue_white_light_blue_cmap", colors)

        # Calculate correlations
        corr = data.corr()

        # Plot heatmap with full matrix visible
        sns.heatmap(corr, annot=True, fmt='.2f',
                    cmap=cmap, center=0, vmin=-1, vmax=1, # Ensure center is at 0
                    linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)

        # Improve readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        st.pyplot(corr_fig)

    # Prediction interface
    st.sidebar.header("Feature Input")
    st.sidebar.info("Adjust the sliders to input feature values")

    # Create sliders for each feature in two columns
    col1, col2 = st.sidebar.columns(2)
    input_features = {}

    for i, feature in enumerate(HIGHLY_CORR_FEATURES):
        if feature in X.columns:
            # Alternate between columns for better layout
            if i % 2 == 0:
                with col1:
                    # CORRECTED LINE: Removed .data.data.frame()
                    input_features[feature] = st.slider(
                        f"{feature}",
                        # CORRECTED LINE: Removed .data.data.frame()
                        min_value=float(X[feature].min()),
                        max_value=float(X[feature].max()),
                        value=float(X[feature].median()),
                        step=0.01
                    )
            else:
                with col2:
                    # CORRECTED LINE: Removed .data.data.frame()
                    input_features[feature] = st.slider(
                        f"{feature}",
                        # CORRECTED LINE: Removed .data.data.frame()
                        min_value=float(X[feature].min()),
                        max_value=float(X[feature].max()),
                        value=float(X[feature].median()),
                        step=0.01
                    )

    if st.sidebar.button("Predict Diagnosis"):
        try:
            # Load models if not already trained
            models = {}
            for model_file in os.listdir(MODEL_SAVE_PATH):
                if model_file.endswith('.pkl') and model_file != 'scaler.pkl':
                    model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
                    with open(os.path.join(MODEL_SAVE_PATH, model_file), 'rb') as f:
                        # CORRECTED LINE: Removed .data.data.frame()
                        models[model_name] = pickle.load(f)

            # Load scaler
            with open(os.path.join(MODEL_SAVE_PATH, 'scaler.pkl'), 'rb') as f:
                scaler = pickle.load(f)

            # Prepare input data
            input_data = pd.DataFrame([input_features])
            input_scaled = scaler.transform(input_data)

            # Make predictions
            st.subheader("Prediction Results")

            # Create a results table
            results_data = []
            for name, model in models.items():
                prediction = model.predict(input_scaled)[0]
                # CORRECTED LINE: Removed .data.data.frame()
                proba = model.predict_proba(input_scaled)[0]

                diagnosis = "Malignant" if prediction == 1 else "Benign"
                # CORRECTED LINE: Removed .data.data.frame()
                confidence = proba[1] if prediction == 1 else proba[0]

                results_data.append({
                    "Model": name,
                    "Diagnosis": diagnosis,
                    "Confidence": f"{confidence:.1%}",
                    "Malignant Probability": f"{proba[1]:.1%}",
                    "Benign Probability": f"{proba[0]:.1%}"
                })

            # Convert to DataFrame and display
            # CORRECTED LINE: Removed .data.data.frame()
            results_df = pd.DataFrame(results_data)

            # Style the DataFrame
            def color_diagnosis(val):
                color = 'red' if val == 'Malignant' else 'green'
                return f'color: {color}; font-weight: bold'

            styled_df = results_df.style.applymap(color_diagnosis, subset=['Diagnosis'])

            # Display in the center
            st.table(styled_df)

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please train the models first by clicking the 'Train All Models' button.")


if __name__ == "__main__":
    main()
