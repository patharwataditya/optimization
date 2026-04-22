"""Neural network training with optimizer comparison."""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam, Adagrad
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from config import (NN_EPOCHS, NN_BATCH_SIZE, NN_VALIDATION_SPLIT, 
                    NN_EARLY_STOPPING_PATIENCE, RANDOM_STATE)


def set_seeds():
    """Set seeds for reproducibility."""
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)


def create_neural_network(input_dim):
    """
    Create a neural network model.
    
    Args:
        input_dim (int): Number of input features.
        
    Returns:
        keras.Model: Compiled neural network model.
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    return model


def train_with_optimizer(X_train, y_train, optimizer_name, optimizer):
    """
    Train neural network with a specific optimizer.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training targets.
        optimizer_name (str): Name of the optimizer.
        optimizer: Keras optimizer object.
        
    Returns:
        tuple: (model, history, best_epoch)
    """
    print(f"Training with {optimizer_name} optimizer...")
    
    # Set seeds for reproducibility
    set_seeds()
    
    # Create model
    model = create_neural_network(X_train.shape[1])
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='val_auc',
        patience=NN_EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        mode='max'
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=NN_EPOCHS,
        batch_size=NN_BATCH_SIZE,
        validation_split=NN_VALIDATION_SPLIT,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Find best epoch (highest validation AUC)
    best_epoch = np.argmax(history.history['val_auc']) + 1
    best_val_acc = history.history['val_accuracy'][best_epoch - 1]
    best_val_auc = history.history['val_auc'][best_epoch - 1]
    
    print(f"{optimizer_name} - Best validation accuracy: {best_val_acc:.4f}")
    print(f"{optimizer_name} - Best validation AUC: {best_val_auc:.4f}")
    print(f"{optimizer_name} - Epochs run: {len(history.history['loss'])}")
    
    return model, history, best_epoch


def compare_optimizers(X_train, y_train):
    """
    Compare neural network performance with different optimizers.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training targets.
        
    Returns:
        dict: Results for each optimizer.
    """
    print("="*60)
    print("NEURAL NETWORK OPTIMIZER COMPARISON")
    print("="*60)
    
    # Define optimizers
    optimizers = {
        'SGD': SGD(learning_rate=0.01, momentum=0.9),
        'Adam': Adam(learning_rate=0.001),
        'Adagrad': Adagrad(learning_rate=0.01)
    }
    
    results = {}
    histories = {}
    
    # Train with each optimizer
    for name, optimizer in optimizers.items():
        model, history, best_epoch = train_with_optimizer(X_train, y_train, name, optimizer)
        results[name] = {
            'model': model,
            'val_accuracy': history.history['val_accuracy'][best_epoch - 1],
            'val_auc': history.history['val_auc'][best_epoch - 1],
            'epochs_run': len(history.history['loss']),
            'best_epoch': best_epoch
        }
        histories[name] = history
    
    # Plot training vs validation loss curves
    plt.figure(figsize=(12, 8))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    for name, history in histories.items():
        plt.plot(history.history['loss'], label=f'{name} - Training')
    plt.title('Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    for name, history in histories.items():
        plt.plot(history.history['val_loss'], label=f'{name} - Validation')
    plt.title('Model Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(2, 2, 3)
    for name, history in histories.items():
        plt.plot(history.history['accuracy'], label=f'{name} - Training')
    plt.title('Model Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    for name, history in histories.items():
        plt.plot(history.history['val_accuracy'], label=f'{name} - Validation')
    plt.title('Model Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle('Neural Network Optimizer Comparison')
    plt.tight_layout()
    plt.savefig('plots/nn_optimizer_comparison.png')
    plt.close()
    
    # Print comparison table
    print(f"{'Optimizer':<15} {'Val Accuracy':<15} {'Val AUC':<10} {'Epochs Run':<10}")
    print("-" * 55)
    for name, result in results.items():
        print(f"{name:<15} {result['val_accuracy']:<15.4f} {result['val_auc']:<10.4f} {result['epochs_run']:<10}")
    
    # Find best optimizer
    best_optimizer = max(results.keys(), key=lambda k: results[k]['val_auc'])
    print(f"\nBest optimizer: {best_optimizer} (AUC: {results[best_optimizer]['val_auc']:.4f})")
    
    return results, best_optimizer


def evaluate_best_nn(results, best_optimizer, X_test, y_test):
    """
    Evaluate the best neural network on test set.
    
    Args:
        results (dict): Results from optimizer comparison.
        best_optimizer (str): Name of the best optimizer.
        X_test (np.array): Test features.
        y_test (np.array): Test targets.
        
    Returns:
        dict: Test metrics.
    """
    print("="*60)
    print(f"EVALUATING BEST NEURAL NETWORK ({best_optimizer})")
    print("="*60)
    
    # Get best model
    best_model = results[best_optimizer]['model']
    
    # Predict on test set
    test_pred_proba = best_model.predict(X_test).flatten()
    test_predictions = (test_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(y_test, test_predictions),
        'precision': precision_score(y_test, test_predictions, average='macro'),
        'recall': recall_score(y_test, test_predictions, average='macro'),
        'f1_score': f1_score(y_test, test_predictions, average='macro'),
        'auc_roc': roc_auc_score(y_test, test_pred_proba)
    }
    
    # Print metrics
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1-Score: {metrics['f1_score']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test AUC-ROC: {metrics['auc_roc']:.4f}")
    
    return metrics


if __name__ == "__main__":
    # This section would run if we executed this file directly
    print("Neural network module ready.")