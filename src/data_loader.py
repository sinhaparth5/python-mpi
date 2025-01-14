import pandas as pd
import numpy as np
from typing import Tuple
import logging
from .utils import setup_logging
from sklearn.preprocessing import StandardScaler
import os

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logging("DataLoader")
        
    def load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess data from the configured data path.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels (if available)
        """
        try:
            dataset_config = self.config.datasets[dataset_name]
            self.logger.info(f"Loading {dataset_name} dataset from {dataset_config.path}")
            file_path = dataset_config.path
            
             # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
            self.logger.info(f"Loading {dataset_name} dataset from {file_path}")
            
            df = pd.read_csv(file_path)
            
            # Basic preprocessing
            df = df.dropna()
            
            # Separate features and target
            feature_columns = [col for col in df.columns if col != 'target']
            X = df[feature_columns].values
            y = df['target'].values if 'target' in df.columns else None
            
            # Standardize the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            self.logger.info(f"Successfully loaded {dataset_name} dataset with shape {X_scaled.shape}")
            return X_scaled, y
            
        except Exception as e:
            self.logger.error(f"Error loading {dataset_name} dataset: {str(e)}")
            raise
