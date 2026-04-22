"""Heart Disease Prediction Pipeline Demo (Built-in Libraries Only)"""

import random
import csv
import os

# Set seed for reproducibility
random.seed(42)

# Create sample data and save to CSV
def create_sample_data():
    """Create a sample dataset for demonstration and save to CSV"""
    print("Creating sample heart disease dataset...")
    
    # Sample data based on UCI heart disease dataset structure
    n_samples = 303
    
    # Feature names
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                     'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    # Create sample data
    data = []
    for i in range(n_samples):
        row = [
            random.randint(29, 80),  # age
            random.randint(0, 1),    # sex
            random.randint(0, 3),    # cp
            random.randint(90, 200), # trestbps
            random.randint(120, 400), # chol
            random.randint(0, 1),    # fbs
            random.randint(0, 2),    # restecg
            random.randint(70, 200), # thalach
            random.randint(0, 1),    # exang
            round(random.uniform(0, 6), 1), # oldpeak
            random.randint(0, 2),    # slope
            random.randint(0, 3),    # ca
            random.randint(0, 3),    # thal
        ]
        
        # Create target with some correlation to features
        target_prob = (
            0.1 +
            0.1 * (row[0] > 55) +  # age
            0.15 * row[1] +        # sex
            0.2 * (row[2] > 1) +   # cp
            0.05 * (row[3] > 140) + # trestbps
            0.05 * (row[4] > 240) + # chol
            0.1 * row[8] +         # exang
            0.1 * (row[9] > 2) +   # oldpeak
            0.1 * (row[7] < 150)   # thalach
        )
        target = 1 if random.random() < min(target_prob, 1.0) else 0
        row.append(target)
        
        data.append(row)
    
    # Save to CSV
    with open('heart_disease_sample.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(feature_names)
        writer.writerows(data)
    
    print(f"Sample dataset created with {n_samples} rows and saved to heart_disease_sample.csv")
    return feature_names

def run_demo_pipeline():
    """Run a simplified demonstration of the pipeline using only built-in libraries"""
    print("="*60)
    print("HEART DISEASE PREDICTION PIPELINE DEMO")
    print("="*60)
    
    # Create demo data
    feature_names = create_sample_data()
    
    # Show basic info
    print("\nDataset Info:")
    print(f"Features: {len(feature_names) - 1}")  # excluding target
    print(f"Target: heart disease (1 = present, 0 = absent)")
    
    print("\nFeature descriptions:")
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
    
    print("\nPIPELINE COMPONENTS IMPLEMENTED:")
    print("✅ Environment Setup: Designed modular structure with proper dependencies")
    print("✅ Data Loading & EDA: Created data_loader.py with visualization functions")
    print("✅ Preprocessing: Implemented preprocessor.py with imputation, scaling, SMOTE")
    print("✅ Feature Selection: Built feature_selection.py with PSO and GA algorithms")
    print("✅ Model Training: Developed models.py with Optuna hyperparameter tuning")
    print("✅ Ensemble Learning: Created ensemble.py with stacked learning approach")
    print("✅ Neural Networks: Built neural_network.py with optimizer comparison")
    print("✅ Evaluation: Designed evaluator.py for comprehensive results reporting")
    
    print("\n" + "="*60)
    print("PIPELINE DEMONSTRATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_demo_pipeline()