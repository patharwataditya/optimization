"""Extended test pipeline with more features"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
import seaborn as sns

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
    """Preprocessing of the data including scaling"""
    print("Preprocessing data...")
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print(f"Training set size: {X_train_scaled.shape[0]}")
    print(f"Test set size: {X_test_scaled.shape[0]}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def feature_importance_analysis(X_train, y_train):
    """Analyze feature importance using Random Forest"""
    print("\nAnalyzing feature importance...")
    
    # Train Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 5 most important features:")
    print(feature_importance.head())
    
    return feature_importance

def train_models(X_train, y_train, X_test, y_test):
    """Train and evaluate multiple models"""
    print("Training models...")
    
    # Models to train
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        print(f"{name} CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
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
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc
        }
        
        print(f"{name} Test Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC-ROC: {auc_roc:.4f}")
    
    return results

def plot_results(results, y_test, X_test):
    """Plot results and create visualizations"""
    print("\nGenerating plots...")
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Plot 1: Model comparison
    models = list(results.keys())
    f1_scores = [results[model]['f1_score'] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, f1_scores, color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylabel('F1-Score')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png')
    plt.close()
    
    # Plot 2: Confusion matrix for best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
    best_model = results[best_model_name]['model']
    y_pred = best_model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png')
    plt.close()
    
    print("Plots saved to 'plots/' directory")

def run_pipeline():
    """Run the complete pipeline"""
    print("="*60)
    print("HEART DISEASE PREDICTION PIPELINE - EXTENDED TEST")
    print("="*60)
    
    # Load data
    df = load_sample_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Feature importance analysis
    feature_importance = feature_importance_analysis(X_train, y_train)
    
    # Train and evaluate models
    results = train_models(X_train, y_train, X_test, y_test)
    
    # Generate plots
    plot_results(results, y_test, X_test)
    
    # Display final results
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    print(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'Precision':<10} {'Recall':<10} {'AUC-ROC':<10}")
    print("-" * 75)
    for model, metrics in results.items():
        print(f"{model:<20} {metrics['accuracy']:<10.4f} {metrics['f1_score']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['auc_roc']:<10.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
    print(f"\nBest Model: {best_model[0]} (F1-Score: {best_model[1]['f1_score']:.4f})")
    
    # Show classification report for best model
    best_model_obj = best_model[1]['model']
    y_pred = best_model_obj.predict(X_test)
    print(f"\nDetailed Classification Report for {best_model[0]}:")
    print(classification_report(y_test, y_pred))
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nDeliverables created:")
    print("✅ Model comparison plot: plots/model_comparison.png")
    print("✅ Confusion matrix plot: plots/confusion_matrix.png")
    print("✅ Heart disease dataset: heart_disease_sample.csv")
    print("✅ Preprocessing pipeline with feature scaling")
    print("✅ Multiple ML models with evaluation metrics")
    print("✅ Feature importance analysis")

if __name__ == "__main__":
    run_pipeline()