from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

def get_models():
    """Return dictionary of selected models and their parameter grids"""
    models = {
        'XGBoost': {
            'model': XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'learning_rate_param': 'learning_rate'
        },
        'Random Forest Optimized': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [200, 300, 400],
                'max_depth': [None, 10, 15, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            },
            'learning_rate_param': None
        },
        'Neural Network': {
            'model': MLPClassifier(random_state=42, early_stopping=True, max_iter=1000),  # Increased max_iter
            'params': {
                'hidden_layer_sizes': [(100,), (100, 50), (50, 25)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'batch_size': [32, 64]
            },
            'learning_rate_param': 'learning_rate_init'
        },
        'AdaBoost': {
            'model': AdaBoostClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5, 1.0]
                # Removed deprecated 'algorithm' parameter
            },
            'learning_rate_param': 'learning_rate'
        },
        'SVM RBF Optimized': {
            'model': SVC(random_state=42, probability=True),
            'params': {
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
                'kernel': ['rbf']
            },
            'learning_rate_param': None
        },
        'SGD Classifier': {
            'model': SGDClassifier(random_state=42),
            'params': {
                'loss': ['log_loss', 'hinge'],
                'learning_rate': ['constant', 'optimal', 'invscaling'],
                'eta0': [0.001, 0.01, 0.1],
                'alpha': [0.0001, 0.001, 0.01]
            },
            'learning_rate_param': 'learning_rate'
        }
    }

    # Enhanced Stacking classifier with simpler final estimator
    base_estimators = [
        ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ]

    models['Stacking Enhanced'] = {
        'model': StackingClassifier(
            estimators=base_estimators,
            final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),  # Changed to RF for stability
            passthrough=True,
            cv=5
        ),
        'params': {
            'final_estimator__n_estimators': [50, 100],
            'final_estimator__max_depth': [5, 10]
        },
        'learning_rate_param': None
    }

    return models