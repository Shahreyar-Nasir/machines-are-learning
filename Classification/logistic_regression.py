#!/usr/bin/env python3
"""
Logistic Regression for Breast Cancer Classification

This script demonstrates binary classification using logistic regression
to predict whether a breast tumor is malignant or benign based on various
medical measurements.

Dataset: Wisconsin Breast Cancer Dataset
Features: 9 medical measurements including clump thickness, cell uniformity, etc.
Target: Class (2 = benign, 4 = malignant)

Author: Machine Learning Playground
"""

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load_and_explore_data(data_path='breast_cancer.csv'):
    """
    Load the breast cancer dataset and perform basic exploration.
    
    Args:
        data_path (str): Path to the CSV file
        
    Returns:
        tuple: (X, y) features and target arrays
    """
    try:
        # Check if file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")
            
        print(f"Loading dataset from: {data_path}")
        dataset = pd.read_csv(data_path)
        
        print(f"Dataset shape: {dataset.shape}")
        print(f"Dataset columns: {list(dataset.columns)}")
        
        # Separate features and target
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Target classes: {np.unique(y)}")
        
        # Check for missing values
        missing_values = dataset.isnull().sum()
        if missing_values.any():
            print("Missing values found:")
            print(missing_values[missing_values > 0])
        else:
            print("No missing values found.")
            
        return X, y, dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        
    Returns:
        trained classifier
    """
    from sklearn.linear_model import LogisticRegression
    
    print("Training Logistic Regression model...")
    classifier = LogisticRegression(random_state=0, max_iter=1000)
    classifier.fit(X_train, y_train)
    print("Model training completed.")
    
    return classifier

def evaluate_model(classifier, X_test, y_test, X_train, y_train):
    """
    Evaluate the trained model using various metrics.
    
    Args:
        classifier: Trained model
        X_test: Test features
        y_test: Test targets
        X_train: Training features (for cross-validation)
        y_train: Training targets (for cross-validation)
    """
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    from sklearn.model_selection import cross_val_score
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\\nConfusion Matrix:")
    print(cm)
    
    # Visualization of confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Benign (2)', 'Malignant (4)'],
                yticklabels=['Benign (2)', 'Malignant (4)'])
    plt.title("Confusion Matrix - Breast Cancer Classification")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    
    # Cross-validation
    print("\\nPerforming 10-fold cross-validation...")
    cv_scores = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} ({cv_scores.mean() * 100:.2f}%)")
    print(f"Standard Deviation: {cv_scores.std():.4f} ({cv_scores.std() * 100:.2f}%)")
    
    # Plot cross-validation results
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 11), cv_scores * 100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axhline(y=cv_scores.mean() * 100, color='red', linestyle='--', 
                label=f'Mean Accuracy: {cv_scores.mean() * 100:.2f}%', linewidth=2)
    plt.xticks(range(1, 11))
    plt.title("10-Fold Cross-Validation Results")
    plt.xlabel("Fold Number")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    # Detailed classification report
    print("\\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Benign (2)', 'Malignant (4)']))

def main():
    """
    Main function to run the complete logistic regression workflow.
    """
    print("=" * 60)
    print("BREAST CANCER CLASSIFICATION USING LOGISTIC REGRESSION")
    print("=" * 60)
    
    # Load and explore data
    X, y, dataset = load_and_explore_data()
    
    # Split the dataset
    from sklearn.model_selection import train_test_split
    print("\\nSplitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Feature scaling
    from sklearn.preprocessing import StandardScaler
    print("\\nApplying feature scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Feature scaling completed.")
    
    # Train model
    classifier = train_logistic_regression(X_train_scaled, y_train)
    
    # Evaluate model
    print("\\n" + "=" * 40)
    print("MODEL EVALUATION RESULTS")
    print("=" * 40)
    evaluate_model(classifier, X_test_scaled, y_test, X_train_scaled, y_train)
    
    print("\\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()