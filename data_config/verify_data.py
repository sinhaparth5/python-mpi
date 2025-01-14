# data_config/verify_data.py
import os
import pandas as pd

def verify_creditcard_data():
    """Verify that the credit card dataset is properly downloaded and readable"""
    data_path = os.path.join('data', 'creditcard.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        return False
        
    try:
        df = pd.read_csv(data_path)
        print("Credit Card Dataset Information:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("\nFirst few rows:")
        print(df.head())
        print("\nData Types:")
        print(df.dtypes)
        return True
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return False

if __name__ == "__main__":
    verify_creditcard_data()