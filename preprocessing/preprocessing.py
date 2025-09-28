import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from config import settings


def load_data(filepath):
    """Load and preprocess raw data with enhanced cleaning"""
    data = pd.read_csv(filepath)

    # Enhanced data cleaning
    data = data.drop(['id'], axis=1, errors='ignore')

    # Handle missing values if any
    if data.isnull().sum().sum() > 0:
        print("   Missing values found, applying imputation...")
        # For numerical features, use median imputation
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    # Remove duplicate rows
    initial_shape = data.shape[0]
    data = data.drop_duplicates()
    final_shape = data.shape[0]
    if initial_shape != final_shape:
        print(f"   Removed {initial_shape - final_shape} duplicate rows")

    return data


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


# Backward compatibility functions
def prepare_data(data):
    return prepare_data_enhanced(data)


def scale_data(X_train, X_test):
    return scale_data_enhanced(X_train, X_test)


def feature_selection(data):
    """Simple feature selection for compatibility"""
    return data, data.columns.tolist()[:-1]  # Return all features except target