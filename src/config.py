from dataclasses import dataclass
import yaml
import os
from dotenv import load_dotenv

@dataclass
class Config:
    n_clusters: int
    max_iter: int
    tolerance: float
    random_state: int
    data_path: str
    log_level: str
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        load_dotenv()
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Override with environment variables if they exist
        config_dict['n_clusters'] = int(os.getenv('N_CLUSTERS', config_dict['n_clusters']))
        config_dict['max_iter'] = int(os.getenv('MAX_ITER', config_dict['max_iter']))
        config_dict['tolerance'] = int(os.getenv('TOLERANCE', config_dict['tolerance']))
        config_dict['random_state'] = int(os.getenv('RANDOM_STATE', config_dict['random_state']))
        config_dict['data_path'] = int(os.getenv('DATA_PATH', config_dict['data_path']))
        config_dict['log_level'] = int(os.getenv('LOG_LEVEL', config_dict['log_level']))
        
        return cls(**config_dict)
    