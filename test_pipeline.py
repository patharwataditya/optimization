"""Test pipeline with available libraries"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import random
import csv

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def load_sample_data():
    """Load or create sample heart disease data"""
    try:
        # Try to load existing data
        df = pd.read_csv('heart_disease_sample.csv')
        print("Loaded existing heart disease sample data")
    except FileNotFoundError:
        # Create sample data if file doesn't exist
        print("Creating sample heart disease dataset...")
        
        # Sample data based on UCI heart disease dataset structure
        n_samples = 303
        
        # Generate features with realistic ranges
        data = {
            'age': np.random.randint(29, 80, n_samples),
            'sex': np.random.binomial(1, 0.7, n_samples),
            'cp': np.random.randint(0, 4, n_samples),
            'trestbps': np.random.randint(90, 200, n_samples),
            'chol': np.random.randint(120, 400, n_samples),
            'fbs': np.random.binomial(1, 0.15, n_samples),
            'restecg': np.random.randint(0, 3, n_samples),
            'thalach': np.random.randint(70, 200, n_samples),
            'exang': np.random.binomial(1, 0.3, n_samples),
            'oldpeak': np.round(np.random.uniform(0, 6, n_samples), 1),
            'slope': np.random.randint(0, 3, n_samples),
            'ca': np.random.randint(0, 4, n_samples),
            'thal': np.random.randint(0, 4, n_samples)
        }
        
        # Create target with correlation to features
        target_prob = (
            0.1 +
            0.1 * (data['age'] > 55) +
            0.15 * data['sex'] +
            0.2 * (data['cp'] > 1) +
            0.05 * (data['trestbps'] > 140) +
            0.05 * (data['chol'] > 240) +
            0.1 * data['exang'] +
            0.1 * (data['oldpeak'] > 2) +
            0.1 * (data['thalach'] < 150)
        )
        data['target'] = np.random.binomial(1, np.clip(target_prob, 0, 1), n_samples)
        
        df = pd.DataFrame(data)
        df.to_csv('heart_disease_sample.csv', index=False)
        print(f"Sample dataset created with shape: {df.shape}")
    
    return df

def preprocess_data(df):
    """Basic preprocessing of the data"""
    print("Preprocessing data...")
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train, X_test, y_test):
    """Train and evaluate multiple models"""
    print("Training models...")
    
    # Models to train
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc
        }
        
        print(f"{name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC-ROC: {auc_roc:.4f}")
    
    return results

def run_pipeline():
    """Run the complete pipeline"""
    print("="*60)
    print("HEART DISEASE PREDICTION PIPELINE - LIBRARY TEST")
    print("="*60)
    
    # Load data
    df = load_sample_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Train and evaluate models
    results = train_models(X_train, y_train, X_test, y_test)
    
    # Display final results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    print(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'AUC-ROC':<10}")
    print("-" * 50)
    for model, metrics in results.items():
        print(f"{model:<20} {metrics['accuracy']:<10.4f} {metrics['f1_score']:<10.4f} {metrics['auc_roc']:<10.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
    print(f"\nBest Model: {best_model[0]} (F1-Score: {best_model[1]['f1_score']:.4f})")
    
    print("\nPipeline execution completed successfully!")

if __name__ == "__main__":
    run_pipeline()