# src/data_fetcher.py
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_openml, load_digits
import os
from sklearn.model_selection import train_test_split

def fetch_and_save_datasets():
    """Fetch datasets and save them to data directory"""
    os.makedirs('data', exist_ok=True)
    
    print("Fetching Covertype dataset...")
    try:
        # Fetch Covertype
        covertype = fetch_ucirepo(id=31)
        X_cover = covertype.data.features
        y_cover = covertype.data.targets
        
        # Save Covertype
        X_cover.to_csv('data/covertype_features.csv', index=False)
        y_cover.to_csv('data/covertype_target.csv', index=False)
        print("Covertype dataset saved successfully!")
    except Exception as e:
        print(f"Error fetching Covertype dataset: {e}")

    print("\nPreparing MNIST dataset (using digits dataset as fallback)...")
    try:
        # Use smaller digits dataset as fallback
        digits = load_digits()
        X_digits = pd.DataFrame(digits.data)
        y_digits = pd.Series(digits.target)
        
        # Save digits dataset
        X_digits.to_csv('data/mnist_features.csv', index=False)
        y_digits.to_csv('data/mnist_target.csv', index=False)
        print("Digits dataset saved successfully!")
    except Exception as e:
        print(f"Error preparing digits dataset: {e}")

if __name__ == "__main__":
    fetch_and_save_datasets()