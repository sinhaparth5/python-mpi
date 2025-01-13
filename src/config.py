from dataclasses import dataclass
import yaml
import os
from dotenv import load_dotenv
from typing import Dict

@dataclass
class DatasetConfig:
    path: str
    n_clusters: int

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
        datasets = {
            name: DatasetConfig(**cfg)
            for name, cfg in config_dict['datasets'].items()
        }
            
        # Override with environment variables if they exist
        
        return cls(
            datasets=datasets,
            max_iter=int(os.getenv('MAX_ITER', config_dict['max_iter'])),
            tolerance=int(os.getenv('TOLERANCE', config_dict['tolerance'])),
            random_state=int(os.getenv('RANDOM_STATE', config_dict['random_state'])),
            log_level=int(os.getenv('LOG_LEVEL', config_dict['log_level']))
        )
    