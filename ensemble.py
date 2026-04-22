"""Stacked ensemble learning implementation."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, classification_report,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import CV_FOLDS, RANDOM_STATE


def create_stacked_ensemble(base_models, X_train, y_train, X_test):
    """
    Create a stacked ensemble using base models.
    
    Args:
        base_models (dict): Dictionary of trained base models.
        X_train (pd.DataFrame or np.array): Training features.
        y_train (pd.Series or np.array): Training targets.
        X_test (pd.DataFrame or np.array): Test features.
        
    Returns:
        tuple: (meta_model, meta_X_train, meta_X_test, meta_model_predictions)
    """
    print("="*60)
    print("CREATING STACKED ENSEMBLE")
    print("="*60)
    
    # Create meta-features using cross-validation predictions
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # Create empty arrays for meta-features
    meta_X_train = np.zeros((X_train.shape[0], len(base_models)))
    meta_X_test = np.zeros((X_test.shape[0], len(base_models)))
    
    # For each base model, generate out-of-fold predictions for training and test predictions
    for i, (name, model_info) in enumerate(base_models.items()):
        model = model_info['model']
        
        # Out-of-fold predictions for training set
        oof_preds = cross_val_predict(model, X_train, y_train, cv=skf, method='predict_proba')
        meta_X_train[:, i] = oof_preds[:, 1]  # Probability of positive class
        
        # Predictions on test set
        test_preds = model.predict_proba(X_test)
        meta_X_test[:, i] = test_preds[:, 1]  # Probability of positive class
        
        print(f"Generated predictions for {name}")
    
    # Train meta-learner (Logistic Regression)
    meta_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    meta_model.fit(meta_X_train, y_train)
    
    # Predict on test set
    meta_predictions = meta_model.predict(meta_X_test)
    meta_pred_proba = meta_model.predict_proba(meta_X_test)[:, 1]
    
    print("Stacked ensemble created successfully.")
    return meta_model, meta_X_train, meta_X_test, meta_predictions, meta_pred_proba


def evaluate_stacked_ensemble(y_test, meta_predictions, meta_pred_proba):
    """
    Evaluate the stacked ensemble model.
    
    Args:
        y_test (pd.Series or np.array): True test targets.
        meta_predictions (np.array): Predictions from meta-model.
        meta_pred_proba (np.array): Prediction probabilities from meta-model.
        
    Returns:
        dict: Evaluation metrics.
    """
    print("="*60)
    print("EVALUATING STACKED ENSEMBLE")
    print("="*60)
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, meta_predictions),
        'precision': precision_score(y_test, meta_predictions, average='macro'),
        'recall': recall_score(y_test, meta_predictions, average='macro'),
        'f1_score': f1_score(y_test, meta_predictions, average='macro'),
        'auc_roc': roc_auc_score(y_test, meta_pred_proba)
    }
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, meta_predictions))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, meta_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title('Confusion Matrix - Stacked Ensemble')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix_stacked.png')
    plt.close()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, meta_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {metrics["auc_roc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Stacked Ensemble')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/roc_stacked.png')
    plt.close()
    
    # Print metrics
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    
    return metrics


def run_stacked_ensemble(base_models, X_train, y_train, X_test, y_test):
    """
    Run the complete stacked ensemble pipeline.
    
    Args:
        base_models (dict): Dictionary of trained base models.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test targets.
        
    Returns:
        tuple: (meta_model, metrics)
    """
    # Create stacked ensemble
    meta_model, meta_X_train, meta_X_test, meta_predictions, meta_pred_proba = create_stacked_ensemble(
        base_models, X_train, y_train, X_test)
    
    # Evaluate ensemble
    metrics = evaluate_stacked_ensemble(y_test, meta_predictions, meta_pred_proba)
    
    return meta_model, metrics


if __name__ == "__main__":
    # This section would run if we executed this file directly
    print("Ensemble module ready.")