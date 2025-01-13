from mpi4py import MPI
import numpy as np
import pandas as pd
from .config import Config
from .utils import setup_logging
from .data_loader import DataLoader
from .kmeans import MPIKMeans
import os

def process_dataset(dataset_name: str, config, data_loader, rank, logger):
    """Process a single dataset"""
    try:
        X, y = data_loader.load_dataset(dataset_name)
        
        # Initialize and fit model
        dataset_config = config.datasets[dataset_name]
        model = MPIKMeans(config, n_clusters=dataset_config.n_clusters)
        model.fit(X)
        
        if rank == 0:
            # Create results directory
            os.makedirs('results', exist_ok=True)
            
            # Make predictions
            labels = model.predict(X)
            logger.info(f"Clustering completed successfully for {dataset_name} dataset")
            
            # Save results
            np.save(f'results/{dataset_name}_centroids.npy', model.centroids_)
            np.save(f'results/{dataset_name}_labels.npy', labels)
            
            # Save comparison with true labels
            results_df = pd.DataFrame({
                'true_label': y,
                'predicted_label': labels
            })
            results_df.to_csv(f'results/{dataset_name}_results.csv', index=False)
            
    except Exception as e:
        logger.error(f"Error processing {dataset_name} dataset: {str(e)}")
        raise

def main():
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Set up logging 
    logger = setup_logging(f"Main_Process_{rank}")
    
    try:
        # Load configuration
        config = Config.from_yaml('config.yaml')
        
        if rank == 0:
            logger.info("Starting MPI K-Means clustering applications")
        
        # Load data
        data_loader = DataLoader(config)
        
        for dataset_name in config.datasets.keys():
            if rank == 0:
                logger.info(f"\Processing {dataset_name} dataset...")
            process_dataset(dataset_name, config, data_loader, rank, logger)
        
        if rank == 0:
            # Make predictions
            logger.info("All datasets processed successfully")
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        MPI.COMM_WORLD.Abort(1)
        
if __name__ == "__main__":
    main()
        