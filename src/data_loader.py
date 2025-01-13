import pandas as pd
import numpy as np
from typing import Tuple
import logging
from .utils import setup_logging
from sklearn.preprocessing import StandardScaler

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
            df = pd.read_csv(dataset_config.path)
            
            # Basic preprocessing
            df = df.dropna()
            
            # For both Iris and Wine datasets, 'target' is the label column
            X = df.drop('target', axis=1).values
            y = df['target'].values
            
            # Standardize the features
            scaler = StandardScaler
            X = scaler.fit_transform(X)
            
            self.logger.info(f"Successfully loaded {dataset_name} dataset with shape {X.shape}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error loading {dataset_name} dataset: {str(e)}")
            raise
