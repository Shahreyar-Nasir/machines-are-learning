#!/usr/bin/env python3
"""
Machine Learning Utilities

Common utility functions for data processing, visualization, and model evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

def load_dataset(dataset_name):
    """
    Load a dataset by name.
    
    Args:
        dataset_name (str): 'breast_cancer', 'startups', or 'preprocessing'
    
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    dataset_paths = {
        'breast_cancer': 'Classification/breast_cancer.csv',
        'startups': 'Regression/50_Startups.csv',
        'preprocessing': 'Preprocessing/Data.csv'
    }
    
    if dataset_name not in dataset_paths:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_paths.keys())}")
    
    try:
        df = pd.read_csv(dataset_paths[dataset_name])
        print(f"✅ Loaded {dataset_name} dataset: {df.shape}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {dataset_paths[dataset_name]}")

def quick_eda(df, target_column=None):
    """
    Perform quick exploratory data analysis.
    
    Args:
        df (pandas.DataFrame): Dataset to analyze
        target_column (str): Name of target column
    """
    print("📊 QUICK EXPLORATORY DATA ANALYSIS")
    print("=" * 40)
    
    # Basic info
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Data types
    print("\\nData Types:")
    print(df.dtypes.value_counts())
    
    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        print("\\nMissing Values:")
        print(missing[missing > 0])
    else:
        print("\\nNo missing values found! ✅")
    
    # Numerical columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\\nNumerical Features ({len(numeric_cols)}):")
        print(df[numeric_cols].describe())
    
    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"\\nCategorical Features ({len(categorical_cols)}):")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count} unique values")
            if unique_count <= 10:
                print(f"    Values: {df[col].value_counts().to_dict()}")
    
    # Target analysis
    if target_column and target_column in df.columns:
        print(f"\\nTarget Variable ({target_column}):")
        if df[target_column].dtype in ['object', 'category']:
            print(df[target_column].value_counts())
        else:
            print(df[target_column].describe())

def plot_distributions(df, columns=None, figsize=(15, 10)):
    """
    Plot distributions of numerical columns.
    
    Args:
        df (pandas.DataFrame): Dataset
        columns (list): Columns to plot (None for all numerical)
        figsize (tuple): Figure size
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    n_cols = len(columns)
    if n_cols == 0:
        print("No numerical columns to plot")
        return
    
    n_rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, col in enumerate(columns):
        if i < len(axes):
            axes[i].hist(df[col].dropna(), bins=20, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide empty subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, figsize=(10, 8)):
    """
    Plot correlation matrix for numerical features.
    
    Args:
        df (pandas.DataFrame): Dataset
        figsize (tuple): Figure size
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        print("No numerical columns for correlation analysis")
        return
    
    correlation_matrix = numeric_df.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

def evaluate_classification_model(y_true, y_pred, class_names=None):
    """
    Comprehensive evaluation of classification model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes for display
    """
    print("📈 CLASSIFICATION MODEL EVALUATION")
    print("=" * 40)
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\\nConfusion Matrix:")
    print(cm)
    
    # Classification Report
    print("\\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names or ['Class 0', 'Class 1'],
                yticklabels=class_names or ['Class 0', 'Class 1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def plot_learning_curves(estimator, X, y, cv=5, figsize=(10, 6)):
    """
    Plot learning curves to diagnose bias/variance.
    
    Args:
        estimator: ML model
        X: Features
        y: Target
        cv: Cross-validation folds
        figsize: Figure size
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def compare_model_performance(models_results):
    """
    Compare performance of multiple models.
    
    Args:
        models_results (dict): {'model_name': {'accuracy': value, 'precision': value, ...}}
    """
    if not models_results:
        print("No model results to compare")
        return
    
    print("🏆 MODEL COMPARISON")
    print("=" * 40)
    
    df_results = pd.DataFrame(models_results).T
    print(df_results.round(4))
    
    # Plot comparison
    if len(df_results.columns) > 1:
        fig, axes = plt.subplots(1, len(df_results.columns), figsize=(15, 5))
        if len(df_results.columns) == 1:
            axes = [axes]
        
        for i, metric in enumerate(df_results.columns):
            axes[i].bar(df_results.index, df_results[metric], alpha=0.7)
            axes[i].set_title(f'{metric.capitalize()}')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

def save_results(results, filename):
    """
    Save results to a file.
    
    Args:
        results (dict): Results to save
        filename (str): Output filename
    """
    if isinstance(results, dict):
        df = pd.DataFrame([results])
    else:
        df = pd.DataFrame(results)
    
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

# Example usage and demo functions
def demo_breast_cancer_analysis():
    """Demo analysis of breast cancer dataset."""
    print("🔬 BREAST CANCER DATASET DEMO")
    print("=" * 50)
    
    try:
        df = load_dataset('breast_cancer')
        quick_eda(df, target_column='Class')
        plot_distributions(df.iloc[:, 1:-1])  # Exclude ID and target
        plot_correlation_matrix(df.iloc[:, 1:-1])
    except Exception as e:
        print(f"Error in demo: {e}")

if __name__ == "__main__":
    print("🛠️  MACHINE LEARNING UTILITIES")
    print("Available functions:")
    print("- load_dataset(name)")
    print("- quick_eda(df, target_column)")
    print("- plot_distributions(df)")
    print("- plot_correlation_matrix(df)")
    print("- evaluate_classification_model(y_true, y_pred)")
    print("- plot_learning_curves(estimator, X, y)")
    print("- compare_model_performance(results)")
    print()
    print("Running demo...")
    demo_breast_cancer_analysis()