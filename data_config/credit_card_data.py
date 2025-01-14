# data_config/credit_card_data.py
import kagglehub
import shutil
import os
import pandas as pd

def fetch_creditcard_data():
    """Fetch credit card fraud dataset and store in data folder"""
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        print("Downloading Credit Card Fraud dataset...")
        base_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        
        print(f"Base path: {base_path}")
        
        # List all files in the download directory
        if isinstance(base_path, list):
            base_path = base_path[0]  # Take first path if it's a list
            
        files = os.listdir(base_path)
        print(f"Files found: {files}")
        
        # Find the creditcard.csv file
        csv_path = os.path.join(base_path, 'creditcard.csv')
        destination_file = os.path.join('data', 'creditcard.csv')
        
        if os.path.exists(csv_path):
            # Copy file to data directory
            shutil.copy2(csv_path, destination_file)
            print(f"Successfully saved credit card dataset to: {destination_file}")
            
            # Verify the file
            df = pd.read_csv(destination_file)
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
        else:
            print(f"CSV file not found at expected path: {csv_path}")
            print("Searching for CSV files in downloaded directory...")
            
            # Recursive search for CSV files
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if file.endswith('.csv'):
                        csv_path = os.path.join(root, file)
                        print(f"Found CSV file: {csv_path}")
                        shutil.copy2(csv_path, destination_file)
                        print(f"Copied to: {destination_file}")
                        return
                        
            raise FileNotFoundError("Could not find creditcard.csv in downloaded dataset")
        
    except Exception as e:
        print(f"Error downloading credit card dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fetch_creditcard_data()