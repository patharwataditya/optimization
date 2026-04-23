"""Model training with Bayesian optimization using Optuna."""

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import os
from imblearn.over_sampling import SMOTE
from config import (OPTUNA_TRIALS, OPTUNA_SEED, CV_FOLDS, RANDOM_STATE, 
                    NN_EARLY_STOPPING_PATIENCE)


# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Train a model and evaluate its performance.
    
    Args:
        model: Scikit-learn compatible model.
        X_train (pd.DataFrame or np.array): Training features.
        y_train (pd.Series or np.array): Training targets.
        X_test (pd.DataFrame or np.array): Test features.
        y_test (pd.Series or np.array): Test targets.
        model_name (str): Name of the model for saving plots.
        
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Apply SMOTE to training data
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Train model
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1_score': f1_score(y_test, y_pred, average='macro'),
    }
    
    # Add AUC-ROC if probabilities are available
    if y_pred_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {metrics["auc_roc"]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'plots/roc_{model_name.lower().replace(" ", "_")}.png')
        plt.close()
    else:
        metrics['auc_roc'] = None
    
    return metrics, model


def tune_random_forest(X_train, y_train):
    """
    Tune Random Forest using Optuna.
    
    Args:
        X_train (pd.DataFrame or np.array): Training features.
        y_train (pd.Series or np.array): Training targets.
        
    Returns:
        dict: Best parameters and model.
    """
    print("="*60)
    print("TUNING RANDOM FOREST")
    print("="*60)
    
    def objective(trial):
        # Hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'random_state': RANDOM_STATE
        }
        
        # Create model
        model = RandomForestClassifier(**params)
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_macro')
        
        return np.mean(scores)
    
    # Create study and optimize
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    
    print(f"Best Random Forest F1-Score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    return study.best_params


def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate Random Forest model with Optuna tuning.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test targets.
        
    Returns:
        dict: Dictionary containing trained model and its metrics.
    """
    print("="*60)
    print("TRAINING AND EVALUATING MODEL")
    print("="*60)
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    results = {}
    
    # Random Forest
    print("\nRandom Forest")
    rf_params = tune_random_forest(X_train, y_train)
    rf_model = RandomForestClassifier(**rf_params)
    rf_metrics, rf_trained = evaluate_model(rf_model, X_train, y_train, X_test, y_test, "Random Forest")
    results['Random Forest'] = {
        'model': rf_trained,
        'params': rf_params,
        'metrics': rf_metrics
    }
    print(f"Test Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"Test F1-Score: {rf_metrics['f1_score']:.4f}")
    print(f"Test Precision: {rf_metrics['precision']:.4f}")
    print(f"Test Recall: {rf_metrics['recall']:.4f}")
    print(f"Test AUC-ROC: {rf_metrics['auc_roc']:.4f}")
    
    return results


if __name__ == "__main__":
    # This section would run if we executed this file directly
    print("Models module ready.")