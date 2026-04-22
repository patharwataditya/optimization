"""Configuration file for the Heart Disease Prediction Pipeline."""

# Random state for reproducibility
RANDOM_STATE = 42

# Test size for train-test split
TEST_SIZE = 0.2

# Number of folds for cross-validation
CV_FOLDS = 5

# PSO parameters
PSO_N_PARTICLES = 20
PSO_DIMENSIONS = 13
PSO_ITERS = 30
PSO_OPTIONS = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# GA parameters
GA_POPULATION_SIZE = 30
GA_GENERATIONS = 20
GA_CROSSOVER_PROB = 0.7
GA_MUTATION_PROB = 0.2
GA_TOURNAMENT_SIZE = 3

# Optuna parameters
OPTUNA_TRIALS = 100
OPTUNA_SEED = 42

# Neural Network parameters
NN_EPOCHS = 50
NN_BATCH_SIZE = 32
NN_VALIDATION_SPLIT = 0.2
NN_EARLY_STOPPING_PATIENCE = 10

# Feature names for the dataset
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Target column
TARGET_NAME = 'target'

# All column names
COLUMN_NAMES = FEATURE_NAMES + [TARGET_NAME]