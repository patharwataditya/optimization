"""Data loading and exploratory data analysis module."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import COLUMN_NAMES, FEATURE_NAMES, TARGET_NAME


def load_data():
    """
    Load the Cleveland Heart Disease dataset.
    
    Returns:
        pd.DataFrame: The loaded dataset with proper column names.
    """
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    try:
        # Try to load the real dataset first
        df = pd.read_csv('Heart_disease_cleveland_new.csv')
        print("Successfully loaded the Heart Disease Cleveland dataset.")
        print(f"Dataset shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("Heart_disease_cleveland_new.csv not found.")
        print("Using sample data for demonstration.")
        return load_sample_data()


def load_sample_data():
    """
    Create a sample dataset for demonstration purposes.
    In a real implementation, you would load the actual dataset.
    
    Returns:
        pd.DataFrame: A sample dataset with similar structure to the heart disease dataset.
    """
    # Create a synthetic dataset for demonstration
    np.random.seed(42)
    n_samples = 303  # Same as the actual dataset
    
    # Generate features with realistic distributions
    data = {
        'age': np.random.randint(29, 80, n_samples),
        'sex': np.random.binomial(1, 0.7, n_samples),  # 0=female, 1=male
        'cp': np.random.randint(0, 4, n_samples),  # chest pain type
        'trestbps': np.random.randint(90, 200, n_samples),  # resting blood pressure
        'chol': np.random.randint(120, 400, n_samples),  # serum cholesterol
        'fbs': np.random.binomial(1, 0.15, n_samples),  # fasting blood sugar > 120 mg/dl
        'restecg': np.random.randint(0, 3, n_samples),  # resting electrocardiographic results
        'thalach': np.random.randint(70, 200, n_samples),  # maximum heart rate achieved
        'exang': np.random.binomial(1, 0.3, n_samples),  # exercise induced angina
        'oldpeak': np.round(np.random.uniform(0, 6, n_samples), 1),  # ST depression
        'slope': np.random.randint(0, 3, n_samples),  # slope of peak exercise ST segment
        'ca': np.random.randint(0, 4, n_samples),  # number of major vessels
        'thal': np.random.randint(0, 4, n_samples),  # thalassemia
        'target': np.random.randint(0, 2, n_samples)  # heart disease target
    }
    
    df = pd.DataFrame(data)
    return df


def explore_data(df):
    """
    Perform exploratory data analysis on the dataset.
    
    Args:
        df (pd.DataFrame): The dataset to explore.
    """
    print("="*60)
    print("DATA EXPLORATION")
    print("="*60)
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print("\nColumn data types:")
    print(df.dtypes)
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset description:")
    print(df.describe())
    
    print("\nMissing values count:")
    print(df.isnull().sum())
    
    # Binarize target for the actual dataset, values > 0 indicate presence of heart disease
    print(f"\nTarget distribution before binarization:")
    print(df[TARGET_NAME].value_counts().sort_index())
    
    # Binarize target: 0 remains 0 (no disease), values > 0 become 1 (disease present)
    df[TARGET_NAME] = (df[TARGET_NAME] > 0).astype(int)
    
    print(f"\nTarget distribution after binarization:")
    print(df[TARGET_NAME].value_counts().sort_index())
    
    # Plot class distribution
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x=TARGET_NAME)
    plt.title('Class Distribution of Heart Disease')
    plt.xlabel('Heart Disease (0 = Absent, 1 = Present)')
    plt.ylabel('Count')
    
    # Add count labels on bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('plots/class_distribution.png')
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()
    
    # Boxplots for all numeric features vs target
    feature_names = [col for col in df.columns if col != TARGET_NAME]
    n_features = len(feature_names)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(feature_names):
        if i < len(axes):
            sns.boxplot(data=df, x=TARGET_NAME, y=feature, ax=axes[i])
            axes[i].set_title(f'{feature} vs Heart Disease')
    
    # Remove empty subplots
    for i in range(len(feature_names), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Boxplots of Features vs Heart Disease Target', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/boxplots.png')
    plt.close()
    
    print("✅ Data exploration plots saved to 'plots/' directory")


if __name__ == "__main__":
    # This section would run if we executed this file directly
    # For the full pipeline, these functions will be called from main.py
    print("Data loader module ready.")