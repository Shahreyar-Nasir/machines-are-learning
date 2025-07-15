#!/usr/bin/env python3
"""
Data Exploration and Analysis Script

This script provides comprehensive data exploration for the datasets
in the Machine Learning Playground repository.

Usage: python explore_data.py [dataset_name]
Dataset options: breast_cancer, startups, preprocessing
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_PATHS

def explore_breast_cancer():
    """Explore the breast cancer dataset."""
    print("🔬 BREAST CANCER DATASET EXPLORATION")
    print("=" * 50)
    
    data_path = os.path.join('Classification', DATA_PATHS['breast_cancer'])
    
    try:
        df = pd.read_csv(data_path)
        
        # Basic information
        print(f"Dataset shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # Feature information
        print("\\nFeature Information:")
        feature_cols = df.columns[1:-1]  # Exclude ID and target
        for i, col in enumerate(feature_cols, 1):
            print(f"{i:2d}. {col}")
        
        # Target distribution
        target_counts = df['Class'].value_counts().sort_index()
        print(f"\\nTarget Distribution:")
        print(f"Benign (Class 2): {target_counts[2]} samples ({target_counts[2]/len(df)*100:.1f}%)")
        print(f"Malignant (Class 4): {target_counts[4]} samples ({target_counts[4]/len(df)*100:.1f}%)")
        
        # Statistical summary
        print("\\nStatistical Summary (Features):")
        print(df.iloc[:, 1:-1].describe())
        
        # Visualizations
        plt.figure(figsize=(15, 10))
        
        # Target distribution
        plt.subplot(2, 3, 1)
        target_counts.plot(kind='bar', color=['lightblue', 'salmon'])
        plt.title('Target Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['Benign (2)', 'Malignant (4)'], rotation=0)
        
        # Feature correlations
        plt.subplot(2, 3, 2)
        corr_matrix = df.iloc[:, 1:-1].corr()
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True, cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix')
        
        # Feature distributions
        plt.subplot(2, 3, 3)
        df.iloc[:, 1:-1].hist(bins=20, figsize=(12, 8))
        plt.suptitle('Feature Distributions')
        
        plt.tight_layout()
        plt.show()
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}")
        return None

def explore_startups():
    """Explore the startup dataset."""
    print("🚀 STARTUP DATASET EXPLORATION")
    print("=" * 50)
    
    data_path = DATA_PATHS['startups']
    
    try:
        df = pd.read_csv(data_path)
        
        # Basic information
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {list(df.columns[:-1])}")
        print(f"Target: {df.columns[-1]}")
        
        # Statistical summary
        print("\\nStatistical Summary:")
        print(df.describe())
        
        # State distribution
        if 'State' in df.columns:
            print("\\nState Distribution:")
            print(df['State'].value_counts())
        
        # Visualizations
        plt.figure(figsize=(15, 10))
        
        # Profit distribution
        plt.subplot(2, 3, 1)
        plt.hist(df['Profit'], bins=20, alpha=0.7, color='green')
        plt.title('Profit Distribution')
        plt.xlabel('Profit')
        plt.ylabel('Frequency')
        
        # Spending correlations
        plt.subplot(2, 3, 2)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Spending vs Profit Correlations')
        
        # State-wise profit
        if 'State' in df.columns:
            plt.subplot(2, 3, 3)
            df.groupby('State')['Profit'].mean().plot(kind='bar', color='orange')
            plt.title('Average Profit by State')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}")
        return None

def explore_preprocessing_data():
    """Explore the preprocessing example dataset."""
    print("🔧 PREPROCESSING DATASET EXPLORATION")
    print("=" * 50)
    
    data_path = DATA_PATHS['preprocessing_data']
    
    try:
        df = pd.read_csv(data_path)
        
        # Basic information
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Missing values
        missing_values = df.isnull().sum()
        print("\\nMissing Values:")
        for col, missing in missing_values.items():
            print(f"{col}: {missing} ({missing/len(df)*100:.1f}%)")
        
        # Data types
        print("\\nData Types:")
        print(df.dtypes)
        
        # Sample data
        print("\\nFirst 5 rows:")
        print(df.head())
        
        # Country distribution
        if 'Country' in df.columns:
            print("\\nCountry Distribution:")
            print(df['Country'].value_counts())
        
        # Target distribution
        if 'Purchased' in df.columns:
            print("\\nPurchased Distribution:")
            print(df['Purchased'].value_counts())
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}")
        return None

def main():
    """Main function to run data exploration."""
    print("🤖 MACHINE LEARNING PLAYGROUND - DATA EXPLORER")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1].lower()
    else:
        print("Available datasets:")
        print("1. breast_cancer - Breast cancer classification data")
        print("2. startups - Startup profit prediction data")
        print("3. preprocessing - Data preprocessing example")
        
        choice = input("\\nEnter dataset name (or 'all' for all datasets): ").lower()
        dataset_name = choice
    
    datasets = {
        'breast_cancer': explore_breast_cancer,
        'startups': explore_startups,
        'preprocessing': explore_preprocessing_data
    }
    
    if dataset_name == 'all':
        for name, func in datasets.items():
            print("\\n" + "="*60)
            func()
    elif dataset_name in datasets:
        datasets[dataset_name]()
    else:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available options: {list(datasets.keys())}")

if __name__ == "__main__":
    main()