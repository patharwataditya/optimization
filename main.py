"""Main orchestration script for the Heart Disease Prediction Pipeline."""

import numpy as np
import pandas as pd
import random
import os
import sys

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Try to import required libraries and provide installation guidance if missing
def check_and_import_libraries():
    """Check if all required libraries are installed."""
    required_libs = {
        'numpy': "pip install numpy",
        'pandas': "pip install pandas",
        'matplotlib': "pip install matplotlib",
        'seaborn': "pip install seaborn",
        'sklearn': "pip install scikit-learn",
        'xgboost': "pip install xgboost",
        'imblearn': "pip install imbalanced-learn",
        'optuna': "pip install optuna",
        'pyswarms': "pip install pyswarms",
        'deap': "pip install deap",
        'tensorflow': "pip install tensorflow",
    }
    
    missing_libs = []
    
    for lib, install_cmd in required_libs.items():
        try:
            if lib == 'sklearn':
                import sklearn
            elif lib == 'imblearn':
                import imblearn
            else:
                __import__(lib)
        except ImportError:
            missing_libs.append((lib, install_cmd))
    
    if missing_libs:
        print("❌ Missing required libraries:")
        for lib, cmd in missing_libs:
            print(f"   - {lib}: {cmd}")
        print("\nPlease install the missing libraries and rerun the script.")
        sys.exit(1)
    else:
        print("✅ All required libraries are installed.")

# Check libraries before proceeding
check_and_import_libraries()

# Import our modules
from data_loader import load_sample_data, explore_data
from preprocessor import preprocess_pipeline
from feature_selection import pso_feature_selection, ga_feature_selection, compare_feature_selection_methods, apply_feature_selection
from models import train_and_evaluate_models
from ensemble import run_stacked_ensemble
from neural_network import compare_optimizers, evaluate_best_nn
from evaluator import create_results_summary, generate_final_report
from config import FEATURE_NAMES, TARGET_NAME


def main():
    """Main function to run the complete pipeline."""
    print("="*80)
    print("OPTIMIZED HEART DISEASE RISK PREDICTION PIPELINE")
    print("="*80)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # STEP 1: Environment setup is handled by check_and_import_libraries()
    
    # STEP 2: Data Loading and Exploration
    print("\n📥 STEP 2: DATA LOADING AND EXPLORATION")
    df = load_data()  # Load the actual dataset
    explore_data(df)
    print("✅ Step 2 complete.")
    
    # Prepare features and target
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    
    # STEP 3: Data Preprocessing
    print("\n🧼 STEP 3: DATA PREPROCESSING")
    X_train, X_test, y_train, y_test, scaler, smote = preprocess_pipeline(X, y)
    print("✅ Step 3 complete.")
    
    # STEP 4: Metaheuristic Feature Selection
    print("\n🔍 STEP 4: METAHEURISTIC FEATURE SELECTION")
    pso_results = pso_feature_selection(X_train, y_train)
    ga_results = ga_feature_selection(X_train, y_train)
    feature_selection_method, selected_features, selection_score = compare_feature_selection_methods(pso_results, ga_results)
    
    # Apply feature selection
    if feature_selection_method == 'PSO':
        best_mask = pso_results[0]
        selected_indices = np.where(best_mask == 1)[0]
        selected_features = [FEATURE_NAMES[i] for i in selected_indices]
    else:  # GA
        best_mask = ga_results[0]
        selected_indices = np.where(np.array(best_mask) == 1)[0]
        selected_features = [FEATURE_NAMES[i] for i in selected_indices]
    
    X_train_selected, X_test_selected = apply_feature_selection(
        pd.DataFrame(X_train, columns=FEATURE_NAMES), 
        pd.DataFrame(X_test, columns=FEATURE_NAMES), 
        selected_features
    )
    print("✅ Step 4 complete.")
    
    # STEP 5: Model Training with Bayesian Optimization
    print("\n🤖 STEP 5: MODEL TRAINING WITH BAYESIAN OPTIMIZATION")
    model_results = train_and_evaluate_models(X_train_selected, y_train, X_test_selected, y_test)
    print("✅ Step 5 complete.")
    
    # STEP 6: Stacked Ensemble Learning
    print("\nスタッ STEP 6: STACKED ENSEMBLE LEARNING")
    stacked_model, stacked_metrics = run_stacked_ensemble(model_results, X_train_selected, y_train, X_test_selected, y_test)
    print("✅ Step 6 complete.")
    
    # STEP 7: Neural Network Optimizer Comparison
    print("\n🧠 STEP 7: NEURAL NETWORK OPTIMIZER COMPARISON")
    nn_results, best_optimizer = compare_optimizers(X_train_selected.values, y_train.values)
    nn_test_metrics = evaluate_best_nn(nn_results, best_optimizer, X_test_selected.values, y_test.values)
    # Add test metrics to nn_results for consistency
    nn_results[best_optimizer]['accuracy'] = nn_test_metrics['accuracy']
    nn_results[best_optimizer]['f1_score'] = nn_test_metrics['f1_score']
    nn_results[best_optimizer]['precision'] = nn_test_metrics['precision']
    nn_results[best_optimizer]['recall'] = nn_test_metrics['recall']
    nn_results[best_optimizer]['auc_roc'] = nn_test_metrics['auc_roc']
    print("✅ Step 7 complete.")
    
    # STEP 8: Final Results Summary
    print("\n📊 STEP 8: FINAL RESULTS SUMMARY")
    summary_df = create_results_summary(model_results, stacked_metrics, nn_results, best_optimizer, feature_selection_method)
    generate_final_report(summary_df, feature_selection_method, best_optimizer)
    print("✅ Step 8 complete.")
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()