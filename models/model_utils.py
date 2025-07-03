import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, params, learning_rate_param, X_train, y_train, X_test, y_test):
    """Train and evaluate a model with optional hyperparameter tuning"""
    start_time = time.time()

    if params:
        grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_

        # Capture learning rate if applicable
        learning_rate = None
        if learning_rate_param and learning_rate_param in best_params:
            learning_rate = best_params[learning_rate_param]
    else:
        best_model = model
        best_model.fit(X_train, y_train)
        best_params = "Default"
        learning_rate = None

    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)

    metrics = {
        'train_accuracy': accuracy_score(y_train, train_pred),
        'test_accuracy': accuracy_score(y_test, test_pred),
        'training_time': time.time() - start_time,
        'best_params': best_params,
        'learning_rate': learning_rate,
        'model': best_model,
        'classification_report': classification_report(y_test, test_pred, target_names=['Benign', 'Malignant']),
        'confusion_matrix': confusion_matrix(y_test, test_pred)
    }

    return metrics