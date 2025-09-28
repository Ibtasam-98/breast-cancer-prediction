import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from matplotlib.colors import LinearSegmentedColormap
import warnings

warnings.filterwarnings('ignore')

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

# Import from your new modules
try:
    from preprocessing.preprocessing import load_data, prepare_data_enhanced, scale_data_enhanced
    from models.model_definitions import get_models
    from models.model_utils import evaluate_model_enhanced
    from visualization.plots import (
        plot_comprehensive_performance,
        plot_comprehensive_learning_curves,
        plot_confusion_matrices_comprehensive,
        plot_roc_curves_comprehensive
    )
except ImportError as e:
    st.error(f"Error importing modules: {str(e)}")
    st.info("Make sure all the new Python files are in the correct directory structure.")


# Load data
@st.cache_data
def load_cached_data():
    """Load and preprocess the breast cancer dataset"""
    try:
        data = load_data('dataset/breast-cancer.csv')
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


# Plot functions
def plot_confusion_matrix(cm, model_name):
    """Plot a single confusion matrix"""
    fig, ax = plt.subplots(figsize=(6, 4))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'],
                cbar=False, ax=ax)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j + 0.5, i + 0.25, f"{cm[i, j]}",
                    ha='center', va='center', color='black')

    ax.set_title(f'{model_name}', pad=20)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    return fig


def plot_roc_curve_simple(models, X_test, y_test):
    """Plot ROC curves for all models"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--')

    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

    for (name, model), color in zip(models.items(), colors):
        if hasattr(model, "predict_proba"):
            try:
                probas = model.predict_proba(X_test)
                fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
            except Exception as e:
                st.warning(f"Could not generate ROC curve for {name}: {str(e)}")
                continue

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return fig


# Main function
def main():
    st.title("🧠 Breast Cancer Classification")
    st.markdown("""
    This app uses advanced machine learning to classify breast cancer tumors as benign or malignant
    using an enhanced pipeline with multiple optimized models.
    """)

    # Load data
    data = load_cached_data()
    if data is None:
        st.error("Could not load data. Please check if 'dataset/breast-cancer.csv' exists.")
        return

    # Display dataset info
    st.sidebar.header("Dataset Information")
    st.sidebar.write(f"**Shape:** {data.shape}")
    st.sidebar.write(f"**Features:** {len(data.columns) - 1}")
    st.sidebar.write(f"**Benign (0):** {len(data[data['diagnosis'] == 0])}")
    st.sidebar.write(f"**Malignant (1):** {len(data[data['diagnosis'] == 1])}")

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data_enhanced(data)
    X_train_scaled, X_test_scaled, scaler = scale_data_enhanced(X_train, X_test, method='standard')

    # Get model definitions
    model_definitions = get_models()

    # Train models
    if st.button("🚀 Train All Models"):
        st.info("Training enhanced models with hyperparameter tuning... This may take a few minutes.")

        progress_bar = st.progress(0)
        status_text = st.empty()

        results = []
        trained_models = {}

        for i, (name, definition) in enumerate(model_definitions.items()):
            status_text.text(f"Training {name}...")

            try:
                metrics = evaluate_model_enhanced(
                    definition['model'],
                    definition['params'],
                    definition.get('learning_rate_param'),
                    X_train_scaled,
                    y_train,
                    X_test_scaled,
                    y_test,
                    name
                )

                results.append({
                    'Model': name,
                    'Train Accuracy': metrics['train_accuracy'],
                    'Test Accuracy': metrics['test_accuracy'],
                    'Training Time (s)': metrics['training_time'],
                    'Sensitivity': metrics['sensitivity'],
                    'Specificity': metrics['specificity'],
                    'Precision': metrics['precision']
                })

                trained_models[name] = metrics['model']

                # Save model
                with open(os.path.join(MODEL_SAVE_PATH, f"{name.lower().replace(' ', '_')}.pkl"), 'wb') as f:
                    pickle.dump(metrics['model'], f)

            except Exception as e:
                st.error(f"Error training {name}: {str(e)}")
                continue

            progress_bar.progress((i + 1) / len(model_definitions))

        # Save scaler
        with open(os.path.join(MODEL_SAVE_PATH, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

        status_text.text("✅ Training complete!")

        # Show results
        st.subheader("📊 Model Performance Summary")
        results_df = pd.DataFrame(results).sort_values('Test Accuracy', ascending=False)

        # Format the display
        display_df = results_df.copy()
        display_df['Train Accuracy'] = display_df['Train Accuracy'].apply(lambda x: f'{x:.2%}')
        display_df['Test Accuracy'] = display_df['Test Accuracy'].apply(lambda x: f'{x:.2%}')
        display_df['Training Time (s)'] = display_df['Training Time (s)'].apply(lambda x: f'{x:.2f}')
        display_df['Sensitivity'] = display_df['Sensitivity'].apply(lambda x: f'{x:.2%}')
        display_df['Specificity'] = display_df['Specificity'].apply(lambda x: f'{x:.2%}')
        display_df['Precision'] = display_df['Precision'].apply(lambda x: f'{x:.2%}')

        st.dataframe(display_df)

        # Show best model
        best_model_name = results_df.iloc[0]['Model']
        best_test_accuracy = results_df.iloc[0]['Test Accuracy']
        st.success(f"🎯 Best model: **{best_model_name}** with test accuracy of **{best_test_accuracy:.2%}**")

        # Show classification report for best model
        st.subheader(f"📋 Classification Report for {best_model_name}")
        best_model = trained_models[best_model_name]
        y_pred = best_model.predict(X_test_scaled)
        report_text = classification_report(y_test, y_pred, target_names=['Benign', 'Malignant'])
        st.text(report_text)

        # Visualizations
        st.subheader("📈 Visualizations")

        # Performance comparison
        st.write("### Performance Comparison")
        try:
            perf_fig = plot_comprehensive_performance(results)
            st.pyplot(perf_fig)
        except Exception as e:
            st.warning(f"Could not generate performance plot: {str(e)}")

        # Learning curves
        st.write("### Learning Curves")
        try:
            learning_fig = plot_comprehensive_learning_curves(trained_models, X_train_scaled, y_train)
            st.pyplot(learning_fig)
        except Exception as e:
            st.warning(f"Could not generate learning curves: {str(e)}")

        # Confusion matrices
        st.write("### Confusion Matrices")
        try:
            cm_fig = plot_confusion_matrices_comprehensive(trained_models, X_test_scaled, y_test)
            st.pyplot(cm_fig)
        except Exception as e:
            st.warning(f"Could not generate confusion matrices: {str(e)}")

        # ROC curves
        st.write("### ROC Curves")
        try:
            roc_fig = plot_roc_curves_comprehensive(trained_models, X_test_scaled, y_test)
            st.pyplot(roc_fig)
        except Exception as e:
            # Fallback to simple ROC curve
            roc_fig_simple = plot_roc_curve_simple(trained_models, X_test_scaled, y_test)
            st.pyplot(roc_fig_simple)

        # Feature correlations
        st.write("### Feature Correlations")
        try:
            corr_fig, ax = plt.subplots(figsize=(12, 10))

            # Calculate correlations
            corr = data.corr()

            # Define custom colormap
            colors = ["#000080", "lightblue", "white", "lightskyblue"]
            cmap = LinearSegmentedColormap.from_list("dark_blue_white_light_blue_cmap", colors)

            # Plot heatmap
            sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap, center=0,
                        linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)

            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_title('Feature Correlations', fontsize=16, fontweight='bold', pad=20)

            st.pyplot(corr_fig)
        except Exception as e:
            st.warning(f"Could not generate correlation plot: {str(e)}")

    # Prediction interface
    st.sidebar.header("🔮 Prediction Interface")
    st.sidebar.info("Input feature values to get predictions from all trained models")

    # Get feature names (exclude diagnosis)
    feature_names = [col for col in data.columns if col != 'diagnosis']

    # Create input sliders
    input_features = {}
    st.sidebar.write("### Feature Values")

    # Create two columns for better layout
    col1, col2 = st.sidebar.columns(2)

    for i, feature in enumerate(feature_names):
        if i % 2 == 0:
            with col1:
                input_features[feature] = st.slider(
                    f"{feature}",
                    min_value=float(data[feature].min()),
                    max_value=float(data[feature].max()),
                    value=float(data[feature].median()),
                    step=0.01,
                    key=feature
                )
        else:
            with col2:
                input_features[feature] = st.slider(
                    f"{feature}",
                    min_value=float(data[feature].min()),
                    max_value=float(data[feature].max()),
                    value=float(data[feature].median()),
                    step=0.01,
                    key=feature
                )

    if st.sidebar.button("🔍 Predict Diagnosis"):
        try:
            # Load trained models
            models = {}
            for model_file in os.listdir(MODEL_SAVE_PATH):
                if model_file.endswith('.pkl') and model_file != 'scaler.pkl':
                    model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
                    with open(os.path.join(MODEL_SAVE_PATH, model_file), 'rb') as f:
                        models[model_name] = pickle.load(f)

            if not models:
                st.error("No trained models found. Please train models first.")
                return

            # Load scaler
            with open(os.path.join(MODEL_SAVE_PATH, 'scaler.pkl'), 'rb') as f:
                scaler = pickle.load(f)

            # Prepare input data
            input_data = pd.DataFrame([input_features])
            input_scaled = scaler.transform(input_data)

            # Make predictions
            st.subheader("🎯 Prediction Results")

            # Create a results table
            results_data = []
            for name, model in models.items():
                try:
                    prediction = model.predict(input_scaled)[0]
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(input_scaled)[0]
                    else:
                        # For models without predict_proba, use decision function or default values
                        proba = [1.0, 0.0] if prediction == 0 else [0.0, 1.0]

                    diagnosis = "Malignant" if prediction == 1 else "Benign"
                    confidence = proba[1] if prediction == 1 else proba[0]
                    malignant_prob = proba[1]
                    benign_prob = proba[0]

                    results_data.append({
                        "Model": name,
                        "Diagnosis": diagnosis,
                        "Confidence": f"{confidence:.1%}",
                        "Malignant Probability": f"{malignant_prob:.1%}",
                        "Benign Probability": f"{benign_prob:.1%}"
                    })
                except Exception as e:
                    st.warning(f"Could not get prediction from {name}: {str(e)}")
                    continue

            # Convert to DataFrame and display
            if results_data:
                results_df = pd.DataFrame(results_data)

                # Style the DataFrame
                def color_diagnosis(val):
                    color = 'red' if val == 'Malignant' else 'green'
                    return f'color: {color}; font-weight: bold'

                def highlight_confidence(val):
                    try:
                        conf = float(val.strip('%')) / 100
                        if conf > 0.9:
                            return 'background-color: #90EE90'  # Light green
                        elif conf > 0.7:
                            return 'background-color: #FFE4B5'  # Light orange
                        else:
                            return 'background-color: #FFB6C1'  # Light red
                    except:
                        return ''

                styled_df = results_df.style.map(color_diagnosis, subset=['Diagnosis']) \
                    .map(highlight_confidence, subset=['Confidence'])

                st.table(styled_df)

                # Show consensus
                diagnoses = [row['Diagnosis'] for row in results_data]
                malignant_count = diagnoses.count('Malignant')
                benign_count = diagnoses.count('Benign')

                if malignant_count > benign_count:
                    consensus = "Malignant"
                    consensus_color = "red"
                else:
                    consensus = "Benign"
                    consensus_color = "green"

                st.markdown(
                    f"### Consensus Prediction: <span style='color:{consensus_color}; font-weight:bold'>{consensus}</span>",
                    unsafe_allow_html=True)
                st.write(f"**Malignant votes:** {malignant_count}/{len(results_data)}")
                st.write(f"**Benign votes:** {benign_count}/{len(results_data)}")

            else:
                st.error("No models could make predictions.")

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please train the models first by clicking the 'Train All Models' button.")

    # Model information
    st.sidebar.header("📋 Model Information")
    st.sidebar.write(f"**Available Models:** {len(model_definitions)}")
    for i, model_name in enumerate(model_definitions.keys(), 1):
        st.sidebar.write(f"{i}. {model_name}")


if __name__ == "__main__":
    main()