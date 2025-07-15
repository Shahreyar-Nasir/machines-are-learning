#!/usr/bin/env python3
"""
Setup script for Machine Learning Playground

This script helps new users set up their environment and verify that
everything is working correctly.
"""

import subprocess
import sys
import os
import importlib

def check_python_version():
    """Check if Python version is adequate."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is supported")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not supported")
        print("Please upgrade to Python 3.8 or higher")
        return False

def install_requirements():
    """Install required packages."""
    print("\\nInstalling required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        print("Please run: pip install -r requirements.txt")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False

def check_packages():
    """Check if all required packages are available."""
    print("\\nChecking package availability...")
    required_packages = [
        ('pandas', 'pandas'), 
        ('numpy', 'numpy'), 
        ('sklearn', 'scikit-learn'),  # sklearn is the import name for scikit-learn
        ('matplotlib', 'matplotlib'), 
        ('seaborn', 'seaborn'), 
        ('scipy', 'scipy')
    ]
    
    missing_packages = []
    for import_name, display_name in required_packages:
        try:
            importlib.import_module(import_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name}")
            missing_packages.append(display_name)
    
    return len(missing_packages) == 0

def check_datasets():
    """Check if datasets are available."""
    print("\\nChecking datasets...")
    datasets = [
        ("Classification/breast_cancer.csv", "Breast Cancer Dataset"),
        ("Regression/50_Startups.csv", "Startup Dataset"),
        ("Preprocessing/Data.csv", "Preprocessing Dataset")
    ]
    
    all_present = True
    for path, name in datasets:
        if os.path.exists(path):
            print(f"✅ {name}")
        else:
            print(f"❌ {name} (missing: {path})")
            all_present = False
    
    return all_present

def run_sample_test():
    """Run a quick test to ensure everything works."""
    print("\\nRunning sample test...")
    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        
        # Create sample data
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 2, 100)
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        
        print(f"✅ Sample test completed (Test accuracy: {accuracy:.2f})")
        return True
        
    except Exception as e:
        print(f"❌ Sample test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user."""
    print("\\n" + "="*60)
    print("🎉 SETUP COMPLETE! Here's what you can do next:")
    print("="*60)
    print()
    print("1. 📊 Explore the datasets:")
    print("   python explore_data.py breast_cancer")
    print("   python explore_data.py startups")
    print("   python explore_data.py preprocessing")
    print()
    print("2. 🤖 Run machine learning models:")
    print("   cd Classification && python logistic_regression.py")
    print()
    print("3. 📓 Open Jupyter notebooks:")
    print("   jupyter notebook")
    print()
    print("4. 📚 Read the guides:")
    print("   - README.md: Overview and usage")
    print("   - ML_CONCEPTS_GUIDE.md: Detailed explanations")
    print()
    print("5. 🛠 Customize configurations:")
    print("   - Edit config.py to modify settings")
    print()
    print("Happy learning! 🚀")

def main():
    """Main setup function."""
    print("🤖 MACHINE LEARNING PLAYGROUND SETUP")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\\n⚠️  Package installation failed. You may need to install manually.")
    
    # Check packages
    packages_ok = check_packages()
    
    # Check datasets
    datasets_ok = check_datasets()
    
    # Run test
    test_ok = run_sample_test()
    
    # Summary
    print("\\n" + "="*50)
    print("SETUP SUMMARY")
    print("="*50)
    print(f"Python version: {'✅' if True else '❌'}")
    print(f"Required packages: {'✅' if packages_ok else '❌'}")
    print(f"Datasets: {'✅' if datasets_ok else '❌'}")
    print(f"Sample test: {'✅' if test_ok else '❌'}")
    
    if packages_ok and datasets_ok and test_ok:
        print("\\n✅ Everything looks good!")
        print_next_steps()
    else:
        print("\\n⚠️  Some issues were found. Please check the messages above.")
        if not packages_ok:
            print("   - Try: pip install -r requirements.txt")
        if not datasets_ok:
            print("   - Make sure you've cloned the complete repository")

if __name__ == "__main__":
    main()