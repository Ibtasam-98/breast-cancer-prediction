import time
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import warnings


def evaluate_model_enhanced(model, params, learning_rate_param, X_train, y_train, X_test, y_test, model_name):
    """Enhanced model evaluation with detailed terminal output"""
    print(f"\n{'=' * 60}")
    print(f"TRAINING: {model_name}")
    print(f"{'=' * 60}")

    start_time = time.time()

    # Use stratified K-fold for more reliable validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if params:
        print("🔍 Performing hyperparameter tuning...")
        # Use RandomizedSearchCV for faster search
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)

            search = RandomizedSearchCV(
                model, params,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                n_iter=10,  # Reduced for stability
                random_state=42
            )

            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_

        print(f"✅ Best parameters found: {best_params}")

    else:
        print("⚡ Training with default parameters...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            best_model = model
            best_model.fit(X_train, y_train)
        best_params = "Default"

    # Capture learning rate if applicable
    learning_rate = None
    if learning_rate_param and hasattr(best_model, 'get_params'):
        model_params = best_model.get_params()
        learning_rate = model_params.get(learning_rate_param)
        if learning_rate:
            print(f"📊 Learning rate: {learning_rate}")

    # Generate predictions and probabilities
    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    training_time = time.time() - start_time

    # Generate classification report
    class_report = classification_report(y_test, test_pred, target_names=['Benign', 'Malignant'])

    # Generate confusion matrix
    cm = confusion_matrix(y_test, test_pred)

    # Print detailed results to terminal
    print(f"\n📊 PERFORMANCE SUMMARY for {model_name}:")
    print(f"   Training Accuracy:   {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
    print(f"   Testing Accuracy:    {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"   Training Time:       {training_time:.2f} seconds")
    print(f"   Best Parameters:     {best_params}")

    print(f"\n📋 CLASSIFICATION REPORT:")
    print(class_report)

    print(f"🎯 CONFUSION MATRIX:")
    print("     Predicted")
    print("     Benign  Malignant")
    print(f"True Benign    {cm[0, 0]:4d}     {cm[0, 1]:4d}")
    print(f"     Malignant {cm[1, 0]:4d}     {cm[1, 1]:4d}")

    # Calculate additional metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"\n📈 ADDITIONAL METRICS:")
    print(f"   Sensitivity (Recall): {sensitivity:.4f}")
    print(f"   Specificity:          {specificity:.4f}")
    print(f"   Precision:            {precision:.4f}")

    print(f"{'=' * 60}")

    metrics = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'best_params': best_params,
        'learning_rate': learning_rate,
        'model': best_model,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision
    }

    return metrics


def evaluate_model(model, params, learning_rate_param, X_train, y_train, X_test, y_test):
    """Wrapper for backward compatibility"""
    return evaluate_model_enhanced(model, params, learning_rate_param, X_train, y_train, X_test, y_test, "Model")