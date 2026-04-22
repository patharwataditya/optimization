"""Feature selection using Particle Swarm Optimization and Genetic Algorithm."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
import pyswarms as ps
from deap import base, creator, tools, algorithms
import random
from config import (PSO_N_PARTICLES, PSO_DIMENSIONS, PSO_ITERS, PSO_OPTIONS,
                    GA_POPULATION_SIZE, GA_GENERATIONS, GA_CROSSOVER_PROB,
                    GA_MUTATION_PROB, GA_TOURNAMENT_SIZE, 
                    CV_FOLDS, RANDOM_STATE, FEATURE_NAMES)


def evaluate_features(mask, X_train, y_train):
    """
    Evaluate a feature subset using cross-validated F1-score.
    
    Args:
        mask (list): Binary mask indicating selected features (1=selected, 0=not selected).
        X_train (pd.DataFrame or np.array): Training feature matrix.
        y_train (pd.Series or np.array): Training target vector.
        
    Returns:
        float: Mean cross-validated F1-score (macro) or 0.0 if no features selected.
    """
    # Convert mask to boolean array
    selected_features = np.array(mask, dtype=bool)
    
    # If no features selected, return 0
    if not np.any(selected_features):
        return 0.0
    
    # Select features
    X_selected = X_train.iloc[:, selected_features] if hasattr(X_train, 'iloc') else X_train[:, selected_features]
    
    # Initialize classifier
    clf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
    
    # Perform cross-validation
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X_selected, y_train, cv=skf, scoring='f1_macro')
    
    return np.mean(scores)


def pso_feature_selection(X_train, y_train):
    """
    Perform feature selection using Particle Swarm Optimization.
    
    Args:
        X_train (pd.DataFrame or np.array): Training feature matrix.
        y_train (pd.Series or np.array): Training target vector.
        
    Returns:
        tuple: (best_mask, selected_features, best_score)
    """
    print("="*60)
    print("PARTICLE SWARM OPTIMIZATION FOR FEATURE SELECTION")
    print("="*60)
    
    def fitness_function(particles):
        """Fitness function for PSO."""
        # particles shape: (n_particles, dimensions)
        scores = []
        for particle in particles:
            # Binarize particle position using threshold 0.5
            mask = (particle > 0.5).astype(int)
            score = evaluate_features(mask, X_train, y_train)
            # PSO minimizes, so we minimize (1 - F1_Score)
            scores.append(1 - score)
        return np.array(scores)
    
    # Initialize PSO optimizer
    optimizer = ps.single.GlobalBestPSO(
        n_particles=PSO_N_PARTICLES,
        dimensions=PSO_DIMENSIONS,
        options=PSO_OPTIONS
    )
    
    # Run optimization
    print(f"Running PSO with {PSO_ITERS} iterations...")
    best_cost, best_pos = optimizer.optimize(
        fitness_function, 
        iters=PSO_ITERS,
        verbose=False
    )
    
    # Binarize best position
    best_mask = (best_pos > 0.5).astype(int)
    selected_indices = np.where(best_mask == 1)[0]
    selected_features = [FEATURE_NAMES[i] for i in selected_indices]
    best_f1_score = 1 - best_cost  # Convert back to F1-score
    
    print(f"Best PSO feature mask: {best_mask}")
    print(f"Selected features: {selected_features}")
    print(f"PSO best F1-Score: {best_f1_score:.4f}")
    
    return best_mask, selected_features, best_f1_score


def ga_feature_selection(X_train, y_train):
    """
    Perform feature selection using Genetic Algorithm.
    
    Args:
        X_train (pd.DataFrame or np.array): Training feature matrix.
        y_train (pd.Series or np.array): Training target vector.
        
    Returns:
        tuple: (best_mask, selected_features, best_score)
    """
    print("="*60)
    print("GENETIC ALGORITHM FOR FEATURE SELECTION")
    print("="*60)
    
    # Set random seed for reproducibility
    random.seed(RANDOM_STATE)
    
    # Define fitness function for DEAP
    def eval_features(individual):
        score = evaluate_features(individual, X_train, y_train)
        return (score,)  # DEAP expects tuple
    
    # Setup DEAP framework
    # Create fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Gene: 0 or 1 (not selected or selected)
    toolbox.register("attr_bool", random.randint, 0, 1)
    
    # Individual: list of genes
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     toolbox.attr_bool, n=PSO_DIMENSIONS)
    
    # Population: list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register genetic operators
    toolbox.register("evaluate", eval_features)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=GA_MUTATION_PROB)
    toolbox.register("select", tools.selTournament, tournsize=GA_TOURNAMENT_SIZE)
    
    # Create initial population
    population = toolbox.population(n=GA_POPULATION_SIZE)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    print(f"Starting GA with {GA_GENERATIONS} generations...")
    
    # Evolution loop
    for gen in range(GA_GENERATIONS):
        # Select next generation
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < GA_CROSSOVER_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Apply mutation
        for mutant in offspring:
            if random.random() < GA_MUTATION_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population
        population[:] = offspring
        
        # Print progress
        if (gen + 1) % 5 == 0:
            fits = [ind.fitness.values[0] for ind in population]
            print(f"Generation {gen+1}: Max fitness = {max(fits):.4f}")
    
    # Get best individual
    best_ind = tools.selBest(population, 1)[0]
    best_mask = best_ind
    selected_indices = np.where(np.array(best_mask) == 1)[0]
    selected_features = [FEATURE_NAMES[i] for i in selected_indices]
    best_f1_score = best_ind.fitness.values[0]
    
    print(f"Best GA feature mask: {best_mask}")
    print(f"Selected features: {selected_features}")
    print(f"GA best F1-Score: {best_f1_score:.4f}")
    
    return best_mask, selected_features, best_f1_score


def compare_feature_selection_methods(pso_results, ga_results):
    """
    Compare PSO and GA feature selection methods.
    
    Args:
        pso_results (tuple): Results from PSO (mask, features, score).
        ga_results (tuple): Results from GA (mask, features, score).
        
    Returns:
        str: Name of the better method ('PSO' or 'GA').
    """
    pso_mask, pso_features, pso_score = pso_results
    ga_mask, ga_features, ga_score = ga_results
    
    print("="*60)
    print("FEATURE SELECTION COMPARISON")
    print("="*60)
    
    # Print comparison table
    print(f"{'Method':<15} {'Features Selected':<40} {'F1-Score':<10}")
    print("-" * 65)
    print(f"{'PSO':<15} {str(pso_features):<40} {pso_score:<10.4f}")
    print(f"{'GA':<15} {str(ga_features):<40} {ga_score:<10.4f}")
    
    # Select better method
    if pso_score >= ga_score:
        best_method = 'PSO'
        best_features = pso_features
        best_score = pso_score
        print(f"\nPSO selected for downstream modeling (higher F1-Score: {pso_score:.4f})")
    else:
        best_method = 'GA'
        best_features = ga_features
        best_score = ga_score
        print(f"\nGA selected for downstream modeling (higher F1-Score: {ga_score:.4f})")
    
    return best_method, best_features, best_score


def apply_feature_selection(X_train, X_test, selected_features):
    """
    Apply feature selection to train and test sets.
    
    Args:
        X_train (pd.DataFrame): Training feature matrix.
        X_test (pd.DataFrame): Test feature matrix.
        selected_features (list): List of selected feature names.
        
    Returns:
        tuple: (X_train_selected, X_test_selected)
    """
    print(f"Applying feature selection: {len(selected_features)} features selected")
    
    # Select features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    print("Feature selection applied successfully.")
    return X_train_selected, X_test_selected


if __name__ == "__main__":
    # This section would run if we executed this file directly
    print("Feature selection module ready.")