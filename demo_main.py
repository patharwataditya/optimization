"""Simplified Heart Disease Prediction Pipeline Demo"""

import numpy as np
import pandas as pd
import random
import os

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Create sample data
def create_sample_data():
    """Create a sample dataset for demonstration"""
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
    print(f"Sample dataset created with shape: {df.shape}")
    return df

def run_demo_pipeline():
    """Run a simplified demonstration of the pipeline"""
    print("="*60)
    print("HEART DISEASE PREDICTION PIPELINE DEMO")
    print("="*60)
    
    # Create demo data
    df = create_sample_data()
    
    # Show basic statistics
    print("\nDataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    
    print("\nFeature descriptions:")
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                     'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    print("1. age: Age in years")
    print("2. sex: Sex (1 = male; 0 = female)")
    print("3. cp: Chest pain type")
    print("4. trestbps: Resting blood pressure (mm Hg)")
    print("5. chol: Serum cholesterol (mg/dl)")
    print("6. fbs: Fasting blood sugar > 120 mg/dl")
    print("7. restecg: Resting electrocardiograph results")
    print("8. thalach: Maximum heart rate achieved")
    print("9. exang: Exercise induced angina")
    print("10. oldpeak: ST depression induced by exercise")
    print("11. slope: Slope of peak exercise ST segment")
    print("12. ca: Number of major vessels colored")
    print("13. thal: Thalassemia")
    print("14. target: Heart disease (1 = present, 0 = absent)")
    
    # Simulate feature selection results
    print("\n" + "="*40)
    print("SIMULATED FEATURE SELECTION RESULTS")
    print("="*40)
    
    print("Particle Swarm Optimization:")
    print("- Best features: ['cp', 'thalach', 'exang', 'oldpeak', 'ca']")
    print("- F1-Score: 0.8742")
    
    print("\nGenetic Algorithm:")
    print("- Best features: ['sex', 'cp', 'thalach', 'exang', 'oldpeak', 'slope', 'ca']")
    print("- F1-Score: 0.8621")
    
    print("\nSelected PSO for downstream modeling (higher F1-Score)")
    
    # Simulate model training results
    print("\n" + "="*40)
    print("SIMULATED MODEL TRAINING RESULTS")
    print("="*40)
    
    results = {
        'Logistic Regression': {'Accuracy': 0.8672, 'F1-Score': 0.8615, 'Precision': 0.8523, 'Recall': 0.8710, 'AUC-ROC': 0.8672},
        'SVM': {'Accuracy': 0.8541, 'F1-Score': 0.8472, 'Precision': 0.8367, 'Recall': 0.8581, 'AUC-ROC': 0.8539},
        'Random Forest': {'Accuracy': 0.8803, 'F1-Score': 0.8756, 'Precision': 0.8612, 'Recall': 0.8905, 'AUC-ROC': 0.8803},
        'XGBoost': {'Accuracy': 0.8738, 'F1-Score': 0.8689, 'Precision': 0.8576, 'Recall': 0.8806, 'AUC-ROC': 0.8738},
        'Stacked Ensemble': {'Accuracy': 0.8915, 'F1-Score': 0.8872, 'Precision': 0.8743, 'Recall': 0.9005, 'AUC-ROC': 0.8915},
        'Neural Net (Adam)': {'Accuracy': 0.8725, 'F1-Score': 0.8678, 'Precision': 0.8542, 'Recall': 0.8819, 'AUC-ROC': 0.8725}
    }
    
    # Print results table
    print("\nMaster Results Table:")
    print(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'Precision':<10} {'Recall':<10} {'AUC-ROC':<10}")
    print("-" * 75)
    for model, metrics in results.items():
        print(f"{model:<20} {metrics['Accuracy']:<10.4f} {metrics['F1-Score']:<10.4f} {metrics['Precision']:<10.4f} {metrics['Recall']:<10.4f} {metrics['AUC-ROC']:<10.4f}")
    
    # Find best models
    best_individual = max([(k, v['F1-Score']) for k, v in results.items() if k not in ['Stacked Ensemble', 'Neural Net (Adam)']], key=lambda x: x[1])
    best_overall = max(results.items(), key=lambda x: x[1]['F1-Score'])
    
    print(f"\nFeature Selection Method: PSO")
    print(f"Number of Features Selected: 5")
    print(f"Best Individual Model: {best_individual[0]} (F1-Score: {best_individual[1]:.4f})")
    print(f"Best Overall Model: {best_overall[0]} (F1-Score: {best_overall[1]['F1-Score']:.4f})")
    
    # Recommendation
    print("\nRECOMMENDATION FOR CLINICAL DEPLOYMENT:")
    if 'Stacked' in best_overall[0]:
        print("- Stacked Ensemble is recommended for clinical deployment due to its superior performance")
        print("  and ability to leverage strengths of multiple models.")
    else:
        print(f"- {best_overall[0]} is recommended for clinical deployment due to its balance of")
        print("  performance and interpretability.")
    
    print("\n" + "="*60)
    print("PIPELINE DEMONSTRATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_demo_pipeline()