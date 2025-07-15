# 🤖 Machine Learning Playground

A comprehensive collection of machine learning implementations and data preprocessing techniques for educational purposes. This repository contains practical examples of classification, regression, and data preprocessing using popular Python libraries.

## 📁 Repository Structure

```
machines-are-learning/
├── Classification/          # Classification models and examples
│   ├── breast_cancer.csv   # Breast cancer dataset
│   ├── logistic_regression.py
│   └── logistic_regression.ipynb
├── Preprocessing/           # Data preprocessing techniques
│   ├── Data.csv            # Sample dataset
│   ├── data_preprocessing.ipynb
│   └── data_preprocessing_template.ipynb
├── Regression/              # Regression models and examples
│   ├── 50_Startups.csv     # Startup profit dataset
│   └── multiple_linear_regression.ipynb
├── config.py               # Configuration settings
├── explore_data.py         # Data exploration script
├── utils.py                # Machine learning utilities
├── setup.py                # Environment setup script
├── requirements.txt        # Python dependencies
├── ML_CONCEPTS_GUIDE.md   # Detailed ML concepts guide
└── .gitignore             # Git ignore rules
```

## 🚀 Quick Start

### Option 1: Automated Setup
```bash
git clone https://github.com/Shahreyar-Nasir/machines-are-learning.git
cd machines-are-learning
python setup.py
```

### Option 2: Manual Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Shahreyar-Nasir/machines-are-learning.git
cd machines-are-learning
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python -c "import pandas, sklearn, matplotlib; print('✅ All packages installed!')"
```

## 📊 Explore the Data

Use the interactive data explorer to understand the datasets:

```bash
# Explore all datasets
python explore_data.py all

# Explore specific datasets
python explore_data.py breast_cancer
python explore_data.py startups
python explore_data.py preprocessing
```

## 🎯 Usage Examples

### Classification Example
```bash
cd Classification/
python logistic_regression.py
```

**What you'll learn:**
- Binary classification with logistic regression
- Data preprocessing and feature scaling
- Model evaluation with confusion matrix
- Cross-validation techniques
- Performance visualization

### Data Preprocessing Example
```bash
cd Preprocessing/
jupyter notebook data_preprocessing.ipynb
```

**What you'll learn:**
- Handling missing data
- Encoding categorical variables
- Feature scaling techniques
- Train-test splitting best practices

### Regression Example
```bash
cd Regression/
jupyter notebook multiple_linear_regression.ipynb
```

**What you'll learn:**
- Multiple linear regression
- Feature engineering for categorical data
- Model performance evaluation
- Prediction vs actual comparison

### Using Utilities
```python
from utils import *

# Quick data analysis
df = load_dataset('breast_cancer')
quick_eda(df, target_column='Class')

# Visualizations
plot_distributions(df)
plot_correlation_matrix(df)

# Model evaluation
evaluate_classification_model(y_true, y_pred, class_names=['Benign', 'Malignant'])
```

## 📊 Datasets Description

### Breast Cancer Dataset (`Classification/breast_cancer.csv`)
- **Purpose**: Binary classification to predict malignant vs benign tumors
- **Samples**: 683 instances
- **Features**: 9 medical measurements
  - Clump Thickness, Cell Size Uniformity, Cell Shape Uniformity
  - Marginal Adhesion, Single Epithelial Cell Size, Bare Nuclei
  - Bland Chromatin, Normal Nucleoli, Mitoses
- **Target**: Class (2 = benign, 4 = malignant)
- **Source**: Wisconsin Breast Cancer Database

### Startup Dataset (`Regression/50_Startups.csv`)
- **Purpose**: Predict startup profit based on spending patterns
- **Samples**: 50 startups
- **Features**: R&D Spend, Administration, Marketing Spend, State
- **Target**: Profit (continuous value)
- **Use Case**: Business analytics and investment decisions

### Sample Dataset (`Preprocessing/Data.csv`)
- **Purpose**: Demonstrate data preprocessing techniques
- **Samples**: 10 instances (small for educational clarity)
- **Features**: Country, Age, Salary
- **Target**: Purchased (Yes/No)
- **Challenges**: Missing values, categorical encoding

## 🛠 Techniques Covered

### 🎯 Classification
- **Algorithms**: Logistic Regression
- **Preprocessing**: Feature scaling, train-test split
- **Evaluation**: Accuracy, precision, recall, F1-score, confusion matrix
- **Validation**: 10-fold cross-validation
- **Visualization**: Confusion matrix heatmaps, performance plots

### 📈 Regression  
- **Algorithms**: Multiple Linear Regression
- **Feature Engineering**: One-hot encoding for categorical variables
- **Evaluation**: Predicted vs actual comparison
- **Preprocessing**: Categorical encoding, feature selection

### 🔧 Data Preprocessing
- **Missing Data**: Mean/median/mode imputation
- **Categorical Encoding**: One-hot encoding, label encoding
- **Feature Scaling**: Standardization (Z-score), Min-Max scaling
- **Data Splitting**: Proper train-test partitioning
- **Data Validation**: Missing value detection, data type analysis

### 📊 Data Analysis & Visualization
- **Exploratory Data Analysis (EDA)**: Automated data profiling
- **Statistical Analysis**: Descriptive statistics, correlation analysis
- **Visualization**: Distribution plots, correlation heatmaps, performance charts
- **Data Quality**: Missing value analysis, outlier detection

## 🎓 Learning Path

### Beginner Level
1. **Start with**: `setup.py` for environment setup
2. **Explore**: Use `explore_data.py` to understand datasets
3. **Learn**: Read `ML_CONCEPTS_GUIDE.md` for theory
4. **Practice**: Run `Classification/logistic_regression.py`

### Intermediate Level
1. **Experiment**: Modify parameters in `config.py`
2. **Analyze**: Use functions from `utils.py` for custom analysis
3. **Compare**: Try different preprocessing techniques
4. **Validate**: Implement cross-validation strategies

### Advanced Level
1. **Extend**: Add new algorithms to existing frameworks
2. **Optimize**: Tune hyperparameters and compare models
3. **Visualize**: Create custom plots for model interpretation
4. **Deploy**: Prepare models for production use

## 🔧 Configuration

Customize behavior by editing `config.py`:

```python
# Model parameters
MODEL_CONFIG = {
    'logistic_regression': {
        'random_state': 42,  # For reproducibility
        'max_iter': 1000,    # Maximum iterations
        'test_size': 0.25,   # Test set proportion
        'cv_folds': 10       # Cross-validation folds
    }
}

# Visualization settings
PLOT_CONFIG = {
    'figure_size': (10, 6),
    'color_scheme': 'Blues'
}
```

## 📈 Model Performance

### Logistic Regression (Breast Cancer Classification)
- **Test Accuracy**: 94.74%
- **Cross-validation**: 96.87% ± 1.57%
- **Precision**: 94% (Malignant), 95% (Benign)
- **Recall**: 92% (Malignant), 96% (Benign)

### Multiple Linear Regression (Startup Profit Prediction)
- **Model Type**: Linear relationship between spending and profit
- **Features**: Multi-dimensional input with categorical encoding
- **Evaluation**: Predicted vs actual value comparison
- **Use Case**: Investment decision support

## 🧪 Advanced Features

### Automated Data Exploration
```python
python explore_data.py breast_cancer
```
- Generates comprehensive data reports
- Identifies data quality issues
- Creates visualizations automatically
- Provides statistical summaries

### Model Evaluation Utilities
```python
from utils import evaluate_classification_model
evaluate_classification_model(y_true, y_pred, class_names=['Benign', 'Malignant'])
```
- Confusion matrix with visualization
- Detailed classification report
- Performance metrics calculation
- Learning curve analysis

### Custom Preprocessing Pipeline
```python
from utils import quick_eda, plot_correlation_matrix
df = load_dataset('your_dataset')
quick_eda(df, target_column='target')
plot_correlation_matrix(df)
```

## 🔧 Technologies Used

- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations
- **Scikit-learn**: Machine learning algorithms and tools
- **Matplotlib**: Static plotting and visualization
- **Seaborn**: Statistical data visualization
- **Jupyter**: Interactive development environment

## 📚 Learning Resources

### Included Documentation
- **README.md**: This comprehensive guide
- **ML_CONCEPTS_GUIDE.md**: Detailed explanations of ML concepts
- **Code Comments**: Extensive inline documentation
- **Docstrings**: Function-level documentation

### External Resources
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/)
- [Machine Learning Course by Andrew Ng](https://www.coursera.org/learn/machine-learning)
- [Hands-On Machine Learning Book](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Bug Reports**: Found an issue? Open a GitHub issue
2. **Feature Requests**: Have an idea? We'd love to hear it
3. **Code Contributions**: 
   - Fork the repository
   - Create a feature branch
   - Make your changes
   - Add tests if applicable
   - Submit a pull request

4. **Documentation**: Help improve guides and examples
5. **Datasets**: Suggest additional educational datasets

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🆘 Getting Help

- **Issues**: Check existing [GitHub Issues](https://github.com/Shahreyar-Nasir/machines-are-learning/issues)
- **Setup Problems**: Run `python setup.py` for diagnostics
- **Concepts**: Read `ML_CONCEPTS_GUIDE.md` for detailed explanations
- **Code Questions**: Check function docstrings and comments

## 🎖 Acknowledgments

- Wisconsin Breast Cancer Database contributors
- Scikit-learn development team
- Open source machine learning community
- Educational content inspired by various ML courses and tutorials

---

**Happy Learning! 🚀**

*Built with ❤️ for the machine learning community*