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
        self.scaler = StandardScaler()
        
    def load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess data from the configured data path.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels (if available)
        """
        try:
            dataset_config = self.config.datasets[dataset_name]
            self.logger.info(f"Loading {dataset_name} dataset from {dataset_config.path}")
            
            is_large = dataset_name in ['covertype', 'mnist']
            
            if is_large:
                # Load features and target separately for large datasets
                X = pd.read_csv(dataset_config.path, chunksize=10000)
                y = pd.read_csv(dataset_config.target_path)
                
                # Process chuncks
                X_processed = []
                for chunk in X:
                    chunk_scaled = self.scaler.fit_transform(chunk)
                    X_processed.append(chunk_scaled)
                
                X_scaled = np.vstack(X_processed)
                y = y.values.ravel()
            else:
                df = pd.read_csv(dataset_config.path)
                
                if dataset_name == 'creditcard':
                    X = df.drop(['Class', 'Time'], axis=1).values
                    y = df['Class'].values
                else:
                    feature_cols = [col for col in df.columns if col != 'target']
                    X = df[feature_cols].values
                    y = df['target'].values if 'target' in df.columns else None
                    
                X_scaled = self.scaler.fit_transform(X)
            
            self.logger.info(f"Successsfully loaded {dataset_name} dataset with shape {X_scaled.shape}")
            return X_scaled, y
            
        except Exception as e:
            self.logger.error(f"Error loading {dataset_name} dataset: {str(e)}")
            raise
