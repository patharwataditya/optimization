# Heart Disease Risk Prediction Pipeline

## 📋 Project Overview

This is a complete, end-to-end optimized machine learning pipeline for predicting heart disease risk using the UCI Cleveland Heart Disease dataset. The pipeline incorporates advanced techniques including metaheuristic feature selection, Bayesian hyperparameter optimization, stacked ensemble learning, and neural network optimization.

## 🎯 Purpose

The pipeline aims to:
1. Predict the likelihood of heart disease in patients based on medical indicators
2. Identify the most important diagnostic features for heart disease detection
3. Provide clinicians with a reliable decision-support tool
4. Demonstrate state-of-the-art machine learning techniques in healthcare

## 🧠 Key Features

### 1. Metaheuristic Feature Selection
- **Particle Swarm Optimization (PSO)**: Intelligent swarm-based algorithm to identify optimal feature subsets
- **Genetic Algorithm (GA)**: Evolutionary approach to feature selection
- **Comparative Analysis**: Automatic selection of the best method based on F1-Score

### 2. Advanced Model Training
- **Bayesian Hyperparameter Optimization**: Uses Optuna for intelligent parameter tuning
- **Multiple Algorithms**: Trains and compares 4 different ML models with 100 trials each
- **Cross-Validation**: Ensures robust performance estimates

### 3. Ensemble Learning
- **Stacked Ensemble**: Combines predictions from multiple models
- **Meta-Learning**: Uses logistic regression to optimally weight base models
- **Improved Performance**: Leverages strengths of individual models

### 4. Deep Learning
- **Neural Network Architecture**: Custom deep learning model for complex pattern recognition
- **Optimizer Comparison**: Tests SGD, Adam, and Adagrad optimizers
- **Early Stopping**: Prevents overfitting during training

## 📊 Dataset Information

### Source
- **Dataset**: UCI Cleveland Heart Disease Dataset
- **Records**: 303 patient records
- **Features**: 13 clinical measurements + 1 target variable

### Features
1. **age**: Age in years
2. **sex**: Sex (1 = male; 0 = female)
3. **cp**: Chest pain type (0-3)
4. **trestbps**: Resting blood pressure (mm Hg)
5. **chol**: Serum cholesterol (mg/dl)
6. **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
7. **restecg**: Resting electrocardiographic results (0-2)
8. **thalach**: Maximum heart rate achieved
9. **exang**: Exercise induced angina (1 = yes; 0 = no)
10. **oldpeak**: ST depression induced by exercise relative to rest
11. **slope**: Slope of peak exercise ST segment (0-2)
12. **ca**: Number of major vessels colored by fluoroscopy (0-3)
13. **thal**: Thalassemia (1-3)

### Target Variable
- **target**: Heart disease status
  - 0: No heart disease
  - 1: Heart disease present

## ⚙️ Pipeline Architecture

```
main.py
├── config.py              # Configuration constants
├── data_loader.py         # Data loading and EDA
├── preprocessor.py        # Data preprocessing
├── feature_selection.py   # PSO and GA feature selection
├── models.py              # Optuna-based model tuning
├── ensemble.py            # Stacked ensemble implementation
├── neural_network.py      # Neural network with optimizer comparison
└── evaluator.py           # Results summary and reporting
```

## 🚀 How It Works

### Step 1: Environment Setup
- Verifies all required libraries are installed
- Sets random seeds for reproducibility
- Initializes modular code structure

### Step 2: Data Loading & Exploration
- Loads the UCI heart disease dataset
- Performs comprehensive exploratory data analysis
- Generates visualizations:
  - Class distribution bar chart
  - Feature correlation heatmap
  - Boxplots for all features vs target

### Step 3: Data Preprocessing
- Handles missing values using median imputation
- Applies Min-Max normalization to all features
- Balances classes using SMOTE (training set only)
- Splits data: 80% training, 20% testing

### Step 4: Metaheuristic Feature Selection
- **Particle Swarm Optimization (PSO)**:
  - 20 particles, 30 iterations
  - Dimensions: 13 features
  - Selects optimal feature subset
- **Genetic Algorithm (GA)**:
  - Population: 30, Generations: 20
  - Crossover: Two-point
  - Mutation: Bit flip
- **Comparison**: Selects method with highest F1-Score

### Step 5: Model Training with Bayesian Optimization
All models tuned with Optuna (100 trials each):

- **Logistic Regression**: C, solver
- **Support Vector Machine**: C, gamma, kernel
- **Random Forest**: n_estimators, max_depth, min_samples_split, min_samples_leaf
- **XGBoost**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree

### Step 6: Stacked Ensemble Learning
- Generates out-of-fold predictions with 5-fold CV
- Uses Logistic Regression as meta-learner
- Combines predictions from all base models

### Step 7: Neural Network Optimizer Comparison
- **Network Architecture**:
  - Input → Dense(64) → Dropout(0.3) → Dense(32) → Dropout(0.2) → Dense(1, sigmoid)
- **Optimizers Compared**:
  - SGD with momentum
  - Adam
  - Adagrad
- **Training**: Early stopping with validation split

### Step 8: Final Results Summary
- Comprehensive performance comparison
- Clinical deployment recommendations
- Feature importance analysis

## 📈 Expected Results

### Performance Metrics
With the UCI heart disease dataset, typical results include:

| Model | Accuracy | F1-Score | Precision | Recall | AUC-ROC |
|-------|----------|----------|-----------|--------|---------|
| Logistic Regression | ~0.836 | ~0.836 | ~0.839 | ~0.840 | ~0.933 |
| SVM | ~0.803 | ~0.803 | ~0.806 | ~0.807 | ~0.939 |
| Random Forest | ~0.82+ | ~0.82+ | ~0.82+ | ~0.82+ | ~0.942 |
| XGBoost | ~0.85+ | ~0.85+ | ~0.85+ | ~0.85+ | ~0.945 |
| Stacked Ensemble | ~0.836 | ~0.836 | ~0.845 | ~0.843 | ~0.942 |
| Neural Network | ~0.80+ | ~0.80+ | ~0.80+ | ~0.80+ | ~0.85+ |

### Clinical Insights
- **Best Features**: Typically identifies 8-11 most predictive features
- **Recommendation**: Usually suggests Stacked Ensemble for clinical deployment
- **Performance**: AUC-ROC consistently above 93%, indicating excellent discriminative ability

## 📂 Output Files

### Console Output
- Progress updates for each pipeline step
- Detailed model performance metrics
- Feature selection results
- Clinical deployment recommendations

### Visualization Files (in `plots/` directory)
1. `class_distribution.png` - Target class distribution
2. `correlation_heatmap.png` - Feature correlation matrix
3. `boxplots.png` - Feature distributions by class
4. `roc_*.png` - ROC curves for all models
5. `confusion_matrix_stacked.png` - Confusion matrix for ensemble
6. `nn_optimizer_comparison.png` - Neural network training curves

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/patharwataditya/optimization.git
cd optimization

# Create virtual environment
python -m venv heart_disease_env
source heart_disease_env/bin/activate  # On Windows: heart_disease_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline
```bash
# Activate virtual environment
source heart_disease_env/bin/activate

# Run complete pipeline
python main.py
```

## 🏥 Clinical Applications

### Diagnostic Support
- Assists clinicians in identifying high-risk patients
- Provides quantitative risk assessment
- Helps prioritize follow-up tests

### Feature Importance
- Identifies which clinical measurements are most predictive
- Guides focus during physical examinations
- Informs which tests to prioritize

### Performance Benefits
- Consistent, objective risk assessment
- Reduces inter-clinician variability
- Provides evidence-based decision support

## ⚠️ Important Considerations

### Medical Disclaimer
This tool is intended for research and educational purposes only. It should not be used as a sole determinant for medical diagnosis or treatment decisions. Always consult with qualified healthcare professionals.

### Model Limitations
- Trained on a single dataset (UCI Cleveland)
- May not generalize to other populations
- Performance depends heavily on data quality

### Technical Considerations
- SMOTE applied only to training data to prevent data leakage
- Cross-validation used for unbiased performance estimates
- Proper train/test splits with stratification maintained

## 📚 Technologies Used

### Core Libraries
- **NumPy/Pandas**: Data manipulation and analysis
- **Scikit-learn**: Traditional machine learning algorithms
- **XGBoost**: Gradient boosting implementation
- **Matplotlib/Seaborn**: Data visualization

### Advanced Tools
- **Optuna**: Bayesian hyperparameter optimization
- **PySwarm**: Particle Swarm Optimization implementation
- **DEAP**: Genetic Algorithm framework
- **Imbalanced-learn**: SMOTE implementation
- **TensorFlow/Keras**: Deep learning framework

## 🤝 Contributing

This project is designed to be modular and extensible:
1. Easy to swap datasets
2. Simple to add new models
3. Straightforward to modify algorithms
4. Clear documentation for all components

## 📄 License

This project is released under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the heart disease dataset
- All open-source library contributors
- Medical researchers who made this dataset available

## 📞 Contact

For questions or issues, please open an issue on the GitHub repository.