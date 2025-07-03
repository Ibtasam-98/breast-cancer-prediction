import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import settings


def load_data(filepath):
    """Load and preprocess raw data"""
    data = pd.read_csv(filepath)
    data = data.drop(['id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


def feature_selection(data):
    """Select features based on correlation"""
    corr_matrix = data.corr().abs()

    # Get features highly correlated with target (use a lower threshold if none found)
    target_corr = corr_matrix['diagnosis'].abs().sort_values(ascending=False)
    high_corr_features = target_corr[target_corr > settings.CORRELATION_THRESHOLD_HIGH].index.tolist()

    # If no features meet the high threshold, use top N features
    if not high_corr_features or len(high_corr_features) == 1:  # only contains 'diagnosis'
        print("No features meet high correlation threshold, using top 5 features")
        high_corr_features = target_corr[1:6].index.tolist()  # skip 'diagnosis' and take top 5
    else:
        high_corr_features.remove('diagnosis')  # Remove target variable

    # Remove highly correlated features (with each other)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > settings.CORRELATION_THRESHOLD_HIGH)]
    data = data.drop(to_drop, axis=1, errors='ignore')

    return data, high_corr_features


def prepare_data(data):
    """Split data into features and target, then train-test split"""
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    """Scale features using StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler