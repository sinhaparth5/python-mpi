import pandas as pd
import numpy as np
from typing import Tuple
import logging
from .utils import setup_logging

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logging("DataLoader")
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess data from the configured data path.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels (if available)
        """
        try:
            self.logger.info(f"Loading data from {self.config.data_path}")
            df = pd.read_csv(self.config.data_path)
            
            # Basic preprocessing
            df = df.dropna()
            
            # Assuming last column is the label (if available)
            if 'label' in df.columns:
                X = df.drop('label', axis=1).values
                y = df['label'].values
                return X, y
            else:
                X = df.values
                return X, None
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
