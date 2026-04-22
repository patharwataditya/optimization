# Optimized Heart Disease Risk Prediction

This project implements a complete, end-to-end optimized machine learning pipeline for predicting heart disease risk using the UCI Cleveland Heart Disease dataset. The pipeline incorporates metaheuristic feature selection, stacked ensemble learning, and neural network optimization.

## 📋 Project Overview

- **Dataset**: UCI Cleveland Heart Disease Dataset
- **Target**: Binary classification (1 = disease present, 0 = absent)
- **Methods**: 
  - Feature Selection: Particle Swarm Optimization (PSO) & Genetic Algorithm (GA)
  - Model Tuning: Bayesian Optimization with Optuna
  - Ensemble Learning: Stacked Ensemble with Logistic Regression meta-learner
  - Deep Learning: Neural Network with optimizer comparison

## 🏗️ Pipeline Architecture

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

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- All dependencies listed in `requirements.txt`

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```bash
python main.py
```

## 🔬 Methodology

1. **Environment Setup**: Install and verify all required libraries
2. **Data Loading & Exploration**: Load dataset, perform EDA with visualizations
3. **Preprocessing**: Handle missing values, scale features, apply SMOTE, train-test split
4. **Feature Selection**: Compare PSO vs GA for optimal feature subset
5. **Model Training**: Tune 4 classifiers with Optuna (100 trials each)
6. **Ensemble Learning**: Create stacked ensemble with out-of-fold predictions
7. **Deep Learning**: Compare SGD, Adam, and Adagrad optimizers
8. **Evaluation**: Comprehensive results summary and model comparison

## 📊 Output

All outputs are saved in the `plots/` directory:
- `class_distribution.png`: Target class distribution
- `correlation_heatmap.png`: Feature correlation matrix
- `boxplots.png`: Feature distributions by class
- `roc_*.png`: ROC curves for all models
- `confusion_matrix_stacked.png`: Confusion matrix for stacked ensemble
- `nn_optimizer_comparison.png`: Training curves for NN optimizers

Console output includes:
- Detailed results tables
- Best parameters for all models
- Performance metrics (Accuracy, F1-Score, Precision, Recall, AUC-ROC)
- Model recommendations for clinical deployment

## 📦 Dependencies

See `requirements.txt` for a complete list of dependencies.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.