from dataclasses import dataclass
import yaml
import os
from dotenv import load_dotenv
from typing import Dict, Optional

@dataclass
class DatasetConfig:
    path: str
    n_clusters: int
    target_path: Optional[str] = None
    batch_size: int = 1000

@dataclass
class Config:
    datasets: Dict[str, DatasetConfig]
    max_iter: int
    tolerance: float
    random_state: int
    log_level: str
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        load_dotenv()
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Convert dataset configuration
        datasets = {}
        for name, cfg in config_dict['datasets'].items():
            if 'target_path' not in cfg:
                cfg['target_path'] = None
            if 'batch_size' not in cfg:
                cfg['batch_size'] = 1000
            datasets[name] = DatasetConfig(**cfg)
            
        # Override with environment variables if they exist
        
        return cls(
            datasets=datasets,
            max_iter=int(os.getenv('MAX_ITER', config_dict['max_iter'])),
            tolerance=float(os.getenv('TOLERANCE', config_dict['tolerance'])),
            random_state=int(os.getenv('RANDOM_STATE', config_dict['random_state'])),
            log_level=os.getenv('LOG_LEVEL', config_dict['log_level'])
        )
        
    def get_dataset_config(self, dataset_name: str) -> DatasetConfig:
        """Helper method to get dataset configuration"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found in configuration")
        return self.datasets[dataset_name]
    