"""Data preprocessing module for handling missing values, scaling, and splitting."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from collections import Counter
from config import TEST_SIZE, RANDOM_STATE, CV_FOLDS


def handle_missing_values(X):
    """
    Handle missing values in the dataset using median imputation.
    
    Args:
        X (pd.DataFrame or np.array): Feature matrix with potential missing values.
        
    Returns:
        pd.DataFrame or np.array: Feature matrix with missing values imputed.
    """
    print("Handling missing values...")
    
    # Check for missing values
    missing_count = pd.DataFrame(X).isnull().sum()
    print(f"Missing values per column:\n{missing_count}")
    
    # Apply median imputation
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # If input was a DataFrame, preserve the structure
    if isinstance(X, pd.DataFrame):
        X_imputed = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
    
    print("Missing values handled using median imputation.")
    return X_imputed


def scale_features(X_train, X_test):
    """
    Apply Min-Max normalization to feature matrices.
    
    Args:
        X_train (pd.DataFrame or np.array): Training feature matrix.
        X_test (pd.DataFrame or np.array): Test feature matrix.
        
    Returns:
        tuple: (scaled_X_train, scaled_X_test, scaler)
    """
    print("Scaling features using Min-Max normalization...")
    
    # Initialize the scaler
    scaler = MinMaxScaler()
    
    # Fit on training data and transform both sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # If inputs were DataFrames, preserve the structure
    if isinstance(X_train, pd.DataFrame):
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("Feature scaling complete.")
    return X_train_scaled, X_test_scaled, scaler


def apply_smote(X_train, y_train):
    """
    Apply SMOTE oversampling to handle class imbalance.
    
    Args:
        X_train (pd.DataFrame or np.array): Training feature matrix.
        y_train (pd.Series or np.array): Training target vector.
        
    Returns:
        tuple: (X_resampled, y_resampled, smote_object)
    """
    print("Applying SMOTE to handle class imbalance...")
    
    # Print class distribution before SMOTE
    print(f"Class distribution before SMOTE: {Counter(y_train)}")
    
    # Apply SMOTE
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Print class distribution after SMOTE
    print(f"Class distribution after SMOTE: {Counter(y_resampled)}")
    
    print("SMOTE applied successfully.")
    return X_resampled, y_resampled, smote


def split_data(X, y):
    """
    Split data into train and test sets.
    
    Args:
        X (pd.DataFrame or np.array): Feature matrix.
        y (pd.Series or np.array): Target vector.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("Splitting data into train and test sets...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        stratify=y, 
        random_state=RANDOM_STATE
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Feature count: {X_train.shape[1]}")
    
    # Print class distribution in both sets
    print(f"Training set class distribution: {Counter(y_train)}")
    print(f"Test set class distribution: {Counter(y_test)}")
    
    print("Data split complete.")
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(X, y):
    """
    Full preprocessing pipeline combining all steps.
    
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, smote)
    """
    print("="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60)
    
    # 1. Handle missing values
    X = handle_missing_values(X)
    
    # 2. Split data first to prevent data leakage
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 3. Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # 4. Apply SMOTE only to training set
    X_train_resampled, y_train_resampled, smote = apply_smote(X_train_scaled, y_train)
    
    print("\nPreprocessing pipeline completed.")
    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, scaler, smote


if __name__ == "__main__":
    # This section would run if we executed this file directly
    print("Preprocessor module ready.")