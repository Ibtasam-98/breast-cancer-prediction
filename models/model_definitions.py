from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

def get_models():
    """Return dictionary of models and their parameter grids"""
    models = {
        'SVM Linear': {
            'model': SVC(random_state=42, probability=True),
            'params': {'kernel': ['linear'], 'C': [0.1, 1, 10]},
            'learning_rate_param': None
        },
        'SVM Linear': {
            'model': SVC(random_state=42, probability=True),
            'params': {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
            'learning_rate_param': None
        },
        'SVM RBF': {
            'model': SVC(random_state=42, probability=True),
            'params': {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1]},
            'learning_rate_param': None
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {'n_neighbors': range(3, 15), 'weights': ['uniform', 'distance']},
            'learning_rate_param': None
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 0.5],
                'max_depth': [3, 5]
            },
            'learning_rate_param': 'learning_rate'
        },
        'SGD Classifier': {
            'model': SGDClassifier(random_state=42),
            'params': {
                'loss': ['log_loss', 'hinge'],
                'learning_rate': ['constant', 'optimal', 'invscaling'],
                'eta0': [0.001, 0.01, 0.1]
            },
            'learning_rate_param': 'learning_rate'
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            },
            'learning_rate_param': None
        },
    }

    # Stacking classifier
    estimators = [
        ('svm', SVC(kernel='linear', C=10, probability=True, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ]

    models['Stacking'] = {
        'model': StackingClassifier(estimators=estimators, final_estimator=LogisticRegression()),
        'params': {'final_estimator__C': [0.1, 1, 10]},
        'learning_rate_param': None
    }

    return models