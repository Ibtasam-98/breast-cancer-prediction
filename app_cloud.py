# app.py - Complete Breast Cancer Classification App for Streamlit Cloud
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc, precision_recall_curve)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import warnings
import base64
from io import BytesIO

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def get_models():
    """Return dictionary of selected models and their parameter grids"""
    models = {
        'XGBoost': {
            'model': XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 0.9]
            },
            'learning_rate_param': 'learning_rate'
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'learning_rate_param': None
        },
        'Neural Network': {
            'model': MLPClassifier(random_state=42, early_stopping=True, max_iter=500),
            'params': {
                'hidden_layer_sizes': [(100,), (50, 25)],
                'alpha': [0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]
            },
            'learning_rate_param': 'learning_rate_init'
        },
        'AdaBoost': {
            'model': AdaBoostClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.1, 0.5]
            },
            'learning_rate_param': 'learning_rate'
        },
        'SVM RBF': {
            'model': SVC(random_state=42, probability=True),
            'params': {
                'C': [0.1, 1, 10],
                'gamma': [0.01, 0.1, 'scale']
            },
            'learning_rate_param': None
        }
    }
    return models


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

@st.cache_data
def load_data():
    """Load sample breast cancer dataset"""
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['diagnosis'] = data.target
    # Map to match original: 0=Benign, 1=Malignant
    df['diagnosis'] = df['diagnosis'].map({0: 1, 1: 0})  # Invert to match WDBC convention
    return df


def prepare_data_enhanced(data, test_size=0.2, random_state=42):
    """Enhanced data preparation with stratification"""
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # Use stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def scale_data_enhanced(X_train, X_test, method='standard'):
    """Enhanced scaling with multiple options"""
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'power':
        scaler = PowerTransformer(method='yeo-johnson')
    else:
        scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================

def evaluate_model_enhanced(model, params, learning_rate_param, X_train, y_train, X_test, y_test, model_name):
    """Enhanced model evaluation with detailed output"""
    start_time = time.time()

    # Use stratified K-fold for more reliable validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced for speed

    if params:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)

            search = RandomizedSearchCV(
                model, params,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                n_iter=4,  # Reduced for speed
                random_state=42
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            best_model = model
            best_model.fit(X_train, y_train)
        best_params = "Default"

    # Generate predictions and calculate metrics
    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    training_time = time.time() - start_time

    # Generate classification report and confusion matrix
    class_report = classification_report(y_test, test_pred, target_names=['Benign', 'Malignant'])
    cm = confusion_matrix(y_test, test_pred)

    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    metrics = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'best_params': best_params,
        'model': best_model,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'model_name': model_name
    }
    return metrics


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

# Set style for plots
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Color palette
BLUE_PALETTE = sns.light_palette("#1f77b4", as_cmap=True)
DARK_BLUE = "#1f77b4"
LIGHT_BLUE = "#aec7e8"


def plot_performance_comparison(results):
    """Plot performance comparison across models"""
    df = pd.DataFrame(results).sort_values('test_accuracy', ascending=False)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    # Plot 1: Accuracy Comparison
    x = np.arange(len(df))
    width = 0.35

    ax1.bar(x - width / 2, df['train_accuracy'], width, label='Training', color=DARK_BLUE, alpha=0.7)
    ax1.bar(x + width / 2, df['test_accuracy'], width, label='Testing', color=LIGHT_BLUE, alpha=0.7)
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training vs Testing Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['model_name'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training Time
    ax2.bar(df['model_name'], df['training_time'], color=DARK_BLUE, alpha=0.7)
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Model Training Time')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Additional Metrics
    metrics_data = {
        'Sensitivity': df['sensitivity'],
        'Specificity': df['specificity'],
        'Precision': df['precision']
    }

    x_metrics = np.arange(len(df))
    width_metrics = 0.25

    ax3.bar(x_metrics - width_metrics, df['sensitivity'], width_metrics,
            label='Sensitivity', color='#2ca02c', alpha=0.7)
    ax3.bar(x_metrics, df['specificity'], width_metrics,
            label='Specificity', color='#ff7f0e', alpha=0.7)
    ax3.bar(x_metrics + width_metrics, df['precision'], width_metrics,
            label='Precision', color='#d62728', alpha=0.7)

    ax3.set_xlabel('Models')
    ax3.set_ylabel('Score')
    ax3.set_title('Performance Metrics')
    ax3.set_xticks(x_metrics)
    ax3.set_xticklabels(df['model_name'], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Rank by Test Accuracy
    ranks = np.arange(1, len(df) + 1)
    ax4.barh(ranks, df['test_accuracy'], color=DARK_BLUE, alpha=0.7)
    ax4.set_yticks(ranks)
    ax4.set_yticklabels([f"{i}. {model}" for i, model in enumerate(df['model_name'], 1)])
    ax4.set_xlabel('Test Accuracy')
    ax4.set_title('Model Ranking by Test Accuracy')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_confusion_matrices(models, X_test, y_test):
    """Plot confusion matrices for all models"""
    n_models = len(models)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for i, (name, model) in enumerate(models.items()):
        if i >= len(axes):
            break

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        im = axes[i].imshow(cm_norm, interpolation='nearest', cmap=BLUE_PALETTE, vmin=0, vmax=1)

        # Add text annotations
        for i_arr in range(cm_norm.shape[0]):
            for j_arr in range(cm_norm.shape[1]):
                axes[i].text(j_arr, i_arr, f"{cm[i_arr, j_arr]}\n({cm_norm[i_arr, j_arr]:.2f})",
                             ha="center", va="center", color="black" if cm_norm[i_arr, j_arr] < 0.7 else "white",
                             fontsize=10)

        axes[i].set_title(f'{name}', fontsize=12)
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xticklabels(['Benign', 'Malignant'])
        axes[i].set_yticklabels(['Benign', 'Malignant'])
        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')

    # Hide unused subplots
    for i in range(len(models), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig


def plot_roc_curves(models, X_test, y_test):
    """Plot ROC curves for all models"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')

    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

    for (name, model), color in zip(models.items(), colors):
        if hasattr(model, "predict_proba"):
            try:
                probas = model.predict_proba(X_test)
                fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
                roc_auc = auc(fpr, tpr)

                ax.plot(fpr, tpr, color=color, linewidth=2,
                        label=f'{name} (AUC = {roc_auc:.3f})')
            except Exception as e:
                continue

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - All Models')
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    return fig


def plot_feature_importance(models, feature_names, X_test, y_test):
    """Plot feature importance for tree-based models"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get feature importances from models that support it
    importance_data = []

    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for i, importance in enumerate(importances):
                importance_data.append({
                    'feature': feature_names[i],
                    'importance': importance,
                    'model': model_name
                })

    if importance_data:
        importance_df = pd.DataFrame(importance_data)

        # Get top 10 features by average importance
        top_features = importance_df.groupby('feature')['importance'].mean().nlargest(10).index
        filtered_df = importance_df[importance_df['feature'].isin(top_features)]

        # Create pivot table for plotting
        pivot_df = filtered_df.pivot(index='feature', columns='model', values='importance')

        # Plot
        pivot_df.plot(kind='bar', ax=ax, alpha=0.8)
        ax.set_xlabel('Features')
        ax.set_ylabel('Feature Importance')
        ax.set_title('Top 10 Feature Importance Across Models')
        ax.legend(title='Models')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
    else:
        ax.text(0.5, 0.5, 'No feature importances available\n(only tree-based models support this)',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('Feature Importance')

    plt.tight_layout()
    return fig


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.title("🏥 Breast Cancer Classification Dashboard")
    st.markdown("""
    This interactive app uses machine learning to classify breast cancer tumors as **benign** or **malignant**.
    It trains multiple models, compares their performance, and allows you to make predictions.
    """)

    # Initialize session state for models and data
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'X_test_scaled' not in st.session_state:
        st.session_state.X_test_scaled = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None

    # Load data
    st.sidebar.header("Dataset Information")
    if st.sidebar.button("Load Breast Cancer Dataset"):
        with st.spinner("Loading dataset..."):
            st.session_state.data = load_data()
        st.sidebar.success("Dataset loaded successfully!")

    if st.session_state.data is not None:
        data = st.session_state.data
        st.sidebar.write(f"**Dataset Shape:** {data.shape}")
        st.sidebar.write(f"**Features:** {len(data.columns) - 1}")
        st.sidebar.write(f"**Benign (0):** {len(data[data['diagnosis'] == 0])}")
        st.sidebar.write(f"**Malignant (1):** {len(data[data['diagnosis'] == 1])}")

        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data_enhanced(data)
        X_train_scaled, X_test_scaled, scaler = scale_data_enhanced(X_train, X_test)

        # Store in session state
        st.session_state.scaler = scaler
        st.session_state.X_test_scaled = X_test_scaled
        st.session_state.y_test = y_test

        # Model training section
        st.header("Model Training")

        if st.button("Train All Models"):
            st.info("Training models with hyperparameter tuning... This may take a few minutes.")

            progress_bar = st.progress(0)
            status_text = st.empty()

            model_definitions = get_models()
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

                    results.append(metrics)
                    trained_models[name] = metrics['model']

                except Exception as e:
                    st.error(f"Error training {name}: {str(e)}")
                    continue

                progress_bar.progress((i + 1) / len(model_definitions))

            st.session_state.trained_models = trained_models
            st.session_state.results = results
            status_text.text("Training complete!")

            # Display results
            st.header("Training Results")

            # Create results dataframe
            results_df = pd.DataFrame([{
                'Model': r['model_name'],
                'Train Accuracy': r['train_accuracy'],
                'Test Accuracy': r['test_accuracy'],
                'Training Time (s)': r['training_time'],
                'Sensitivity': r['sensitivity'],
                'Specificity': r['specificity'],
                'Precision': r['precision']
            } for r in results])

            # Sort by test accuracy
            results_df = results_df.sort_values('Test Accuracy', ascending=False)

            # Display formatted table
            display_df = results_df.copy()
            display_df['Train Accuracy'] = display_df['Train Accuracy'].apply(lambda x: f'{x:.2%}')
            display_df['Test Accuracy'] = display_df['Test Accuracy'].apply(lambda x: f'{x:.2%}')
            display_df['Training Time (s)'] = display_df['Training Time (s)'].apply(lambda x: f'{x:.2f}')
            display_df['Sensitivity'] = display_df['Sensitivity'].apply(lambda x: f'{x:.2%}')
            display_df['Specificity'] = display_df['Specificity'].apply(lambda x: f'{x:.2%}')
            display_df['Precision'] = display_df['Precision'].apply(lambda x: f'{x:.2%}')

            st.dataframe(display_df, use_container_width=True)

            # Show best model
            best_model_row = results_df.iloc[0]
            st.success(f" **Best Model:** {best_model_row['Model']} "
                       f"(Test Accuracy: {best_model_row['Test Accuracy']:.2%})")

            # Visualizations
            st.header("Performance Visualizations")

            tab1, tab2, tab3, tab4 = st.tabs([
                "Performance Comparison",
                "Confusion Matrices",
                "ROC Curves",
                "Feature Importance"
            ])

            with tab1:
                st.subheader("Model Performance Comparison")
                perf_fig = plot_performance_comparison(results)
                st.pyplot(perf_fig)

            with tab2:
                st.subheader("Confusion Matrices")
                cm_fig = plot_confusion_matrices(trained_models, X_test_scaled, y_test)
                st.pyplot(cm_fig)

            with tab3:
                st.subheader("ROC Curves")
                roc_fig = plot_roc_curves(trained_models, X_test_scaled, y_test)
                st.pyplot(roc_fig)

            with tab4:
                st.subheader("Feature Importance")
                feature_names = [col for col in data.columns if col != 'diagnosis']
                fi_fig = plot_feature_importance(trained_models, feature_names, X_test_scaled, y_test)
                st.pyplot(fi_fig)

        # Prediction interface
        if st.session_state.trained_models is not None:
            st.header("🔮 Prediction Interface")
            st.markdown("Input feature values to get predictions from all trained models:")

            # Get feature names
            feature_names = [col for col in data.columns if col != 'diagnosis']

            # Create input sliders in columns
            input_features = {}
            cols = st.columns(3)

            for i, feature in enumerate(feature_names):
                col_idx = i % 3
                with cols[col_idx]:
                    input_features[feature] = st.slider(
                        f"{feature}",
                        min_value=float(data[feature].min()),
                        max_value=float(data[feature].max()),
                        value=float(data[feature].median()),
                        step=0.01,
                        key=feature
                    )

            if st.button("Predict Diagnosis"):
                try:
                    # Prepare input data
                    input_data = pd.DataFrame([input_features])
                    input_scaled = st.session_state.scaler.transform(input_data)

                    # Make predictions
                    results_data = []

                    for name, model in st.session_state.trained_models.items():
                        try:
                            prediction = model.predict(input_scaled)[0]
                            if hasattr(model, "predict_proba"):
                                proba = model.predict_proba(input_scaled)[0]
                            else:
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
                            continue

                    # Display results
                    if results_data:
                        results_df = pd.DataFrame(results_data)

                        # Color coding for diagnosis
                        def color_diagnosis(val):
                            color = 'red' if val == 'Malignant' else 'green'
                            return f'color: {color}; font-weight: bold'

                        def highlight_confidence(val):
                            try:
                                conf = float(val.strip('%')) / 100
                                if conf > 0.9:
                                    return 'background-color: #90EE90'
                                elif conf > 0.7:
                                    return 'background-color: #FFE4B5'
                                else:
                                    return 'background-color: #FFB6C1'
                            except:
                                return ''

                        styled_df = results_df.style.map(color_diagnosis, subset=['Diagnosis']) \
                            .map(highlight_confidence, subset=['Confidence'])

                        st.dataframe(styled_df, use_container_width=True)

                        # Show consensus
                        diagnoses = [row['Diagnosis'] for row in results_data]
                        malignant_count = diagnoses.count('Malignant')
                        benign_count = diagnoses.count('Benign')

                        if malignant_count > benign_count:
                            consensus = "Malignant"
                            consensus_color = "red"
                            emoji = "🔴"
                        else:
                            consensus = "Benign"
                            consensus_color = "green"
                            emoji = "🟢"

                        st.markdown(f"### {emoji} Consensus Prediction: "
                                    f"<span style='color:{consensus_color}; font-weight:bold'>{consensus}</span>",
                                    unsafe_allow_html=True)
                        st.write(f"**Malignant votes:** {malignant_count}/{len(results_data)}")
                        st.write(f"**Benign votes:** {benign_count}/{len(results_data)}")

                    else:
                        st.error("No models could make predictions.")

                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

    else:
        st.info("Click 'Load Breast Cancer Dataset' in the sidebar to get started!")

        # Display sample of what the app does
        st.header(" What This App Does")
        st.markdown("""
        This machine learning application:

        - **Loads** the Wisconsin Breast Cancer Dataset
        - **Preprocesses** the data with scaling and feature engineering
        - **Trains** multiple machine learning models:
          - XGBoost
          - Random Forest
          - Neural Network
          - AdaBoost
          - SVM with RBF kernel
        - **Compares** model performance with comprehensive metrics
        - **Visualizes** results with interactive charts
        - **Allows** real-time predictions with new data

        Click the button in the sidebar to load the dataset and begin!
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        "**Breast Cancer Classification App** | "
        "Built with Streamlit, Scikit-learn, and XGBoost"
    )


if __name__ == "__main__":
    main()
