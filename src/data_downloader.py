# data_downloader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine
import os
import logging
from typing import Tuple, Optional
import requests
from pathlib import Path

class DataDownloader:
    def __init__(self):
        self.logger = self._setup_logger()
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('DataDownloader')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger

    def download_iris_data(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Downloads and prepares the Iris dataset
        """
        try:
            self.logger.info("Loading Iris dataset...")
            iris = load_iris()
            X = iris.data
            y = iris.target
            
            # Create DataFrame
            df = pd.DataFrame(X, columns=iris.feature_names)
            df['target'] = y
            
            # Save to CSV
            output_path = 'data/iris_data.csv'
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Iris dataset saved to {output_path}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error downloading Iris dataset: {str(e)}")
            raise

    def download_wine_data(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Downloads and prepares the Wine dataset
        """
        try:
            self.logger.info("Loading Wine dataset...")
            wine = load_wine()
            X = wine.data
            y = wine.target
            
            # Create DataFrame
            df = pd.DataFrame(X, columns=wine.feature_names)
            df['target'] = y
            
            # Save to CSV
            output_path = 'data/wine_data.csv'
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Wine dataset saved to {output_path}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error downloading Wine dataset: {str(e)}")
            raise

    def prepare_mall_customer_data(self, file_path: str) -> np.ndarray:
        """
        Prepares the Mall Customer Segmentation Data.
        Args:
            file_path: Path to the downloaded Mall_Customers.csv file
        """
        try:
            self.logger.info("Preparing Mall Customer dataset...")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Mall Customer data file not found at {file_path}")
            
            # Read the data
            df = pd.read_csv(file_path)
            
            # Select features for clustering
            features = ['Annual Income (k$)', 'Spending Score (1-100)']
            X = df[features]
            
            # Standardize the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Save preprocessed data
            output_path = 'data/mall_customer_data.csv'
            pd.DataFrame(X_scaled, columns=features).to_csv(output_path, index=False)
            
            self.logger.info(f"Mall Customer dataset saved to {output_path}")
            return X_scaled
            
        except Exception as e:
            self.logger.error(f"Error preparing Mall Customer dataset: {str(e)}")
            raise

    def generate_synthetic_data(self, 
                              n_samples: int = 1000, 
                              n_features: int = 2, 
                              n_clusters: int = 3,
                              cluster_std: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates synthetic clustering data
        """
        try:
            self.logger.info("Generating synthetic dataset...")
            from sklearn.datasets import make_blobs
            
            # Generate the blobs
            X, y = make_blobs(n_samples=n_samples,
                            n_features=n_features,
                            centers=n_clusters,
                            cluster_std=cluster_std,
                            random_state=42)
            
            # Create DataFrame
            df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_features)])
            df['target'] = y
            
            # Save to CSV
            output_path = 'data/synthetic_data.csv'
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Synthetic dataset saved to {output_path}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic dataset: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    downloader = DataDownloader()
    
    # Download and prepare different datasets
    try:
        # Iris dataset
        X_iris, y_iris = downloader.download_iris_data()
        print("\nIris dataset shape:", X_iris.shape)
        
        # Wine dataset
        X_wine, y_wine = downloader.download_wine_data()
        print("Wine dataset shape:", X_wine.shape)
        
        # Generate synthetic data
        X_synthetic, y_synthetic = downloader.generate_synthetic_data(
            n_samples=1000,
            n_features=2,
            n_clusters=3
        )
        print("Synthetic dataset shape:", X_synthetic.shape)
        
        # For Mall Customer data, you need to download it first from Kaggle
        # Then you can process it like this:
        # X_mall = downloader.prepare_mall_customer_data('path_to_mall_customers.csv')
        # print("Mall Customer dataset shape:", X_mall.shape)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")