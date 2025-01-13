from mpi4py import MPI
import numpy as np
from .config import Config
from .utils import setup_logging
from .data_loader import DataLoader
from .kmeans import MPIKMeans

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
        X, y = data_loader.load_data()
        
        # Initialize and fit model
        model = MPIKMeans(config)
        model.fit(X)
        
        if rank == 0:
            # Make predictions
            labels = model.predict(X)
            logger.info("Clustering completed successfully")
            
            # Save results if needed
            np.save('results/centroids.npy', model.centroids_)
            np.save('results/labels.npy', labels)
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        MPI.COMM_WORLD.Abort(1)
        
if __name__ == "__main__":
    main()
        