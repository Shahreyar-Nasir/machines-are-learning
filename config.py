"""
Configuration file for Machine Learning Playground

This file contains configurable parameters for the machine learning models.
Modify these settings to customize the behavior of the scripts.
"""

# Data file paths (relative to the script directory)
DATA_PATHS = {
    'breast_cancer': 'breast_cancer.csv',
    'startups': '../Regression/50_Startups.csv',
    'preprocessing_data': '../Preprocessing/Data.csv'
}

# Model parameters
MODEL_CONFIG = {
    'logistic_regression': {
        'random_state': 0,
        'max_iter': 1000,
        'test_size': 0.25,
        'cv_folds': 10
    },
    'linear_regression': {
        'random_state': 0,
        'test_size': 0.2
    }
}

# Visualization settings
PLOT_CONFIG = {
    'figure_size': (10, 6),
    'confusion_matrix_size': (8, 6),
    'color_scheme': 'Blues',
    'dpi': 100
}

# Feature scaling options
SCALING_CONFIG = {
    'method': 'standard',  # 'standard', 'minmax', 'robust'
    'apply_to_target': False
}