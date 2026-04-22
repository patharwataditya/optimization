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
    
    # Since we don't have the actual dataset file, we'll create a function
    # that would load it. In a real scenario, you would have the processed.cleveland.data file.
    print("Note: In a real implementation, this would load processed.cleveland.data")
    print("For now, we'll simulate the data loading process.")
    
    # Returning None to indicate that we need to download the actual dataset
    return None


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
    }
    
    # Create target with some correlation to features
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
    data[TARGET_NAME] = np.random.binomial(1, np.clip(target_prob, 0, 1), n_samples)
    
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
    
    # Binarize target
    print(f"\nTarget distribution before binarization:")
    print(df[TARGET_NAME].value_counts().sort_index())
    
    # For the actual dataset, values > 0 indicate presence of heart disease
    # In our sample data, we already have 0/1 values
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
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for i, feature in enumerate(FEATURE_NAMES):
        if i < len(axes):
            sns.boxplot(data=df, x=TARGET_NAME, y=feature, ax=axes[i])
            axes[i].set_title(f'{feature} vs Heart Disease')
    
    # Remove empty subplots
    for i in range(len(FEATURE_NAMES), len(axes)):
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