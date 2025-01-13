import logging
import os
from typing import Optional

def setup_logging(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the application
    
    Args:
        name (str): Logger name
        level (Optional[str]): Logging level 

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        os.makedirs('logs', exist_ok=True)
        fh = logging.FileHandler(f'logs/{name}.log')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    # Set level from config or default to INFO
    logger.setLevel(level or logging.INFO)
    
    return logger

