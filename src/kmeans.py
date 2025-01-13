import numpy as np
from mpi4py import MPI
import logging
from .utils import setup_logging
from .config import Config

class MPIKMeans:
    def __init__(self, config: Config):
        """
        Initialize the MPI K-means clustering algorithm

        Args:
            config (Config): Configuration object containing parameters
        """
        self.config = config
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.logger = setup_logging(f"MPIKMeans_Process_{self.rank}")
        
        # Initialize parameters from config
        self.n_clusters = config.n_clusters
        self.max_iter = config.max_iter
        self.tol = config.tolerance
        self.random_state = config.random_state
        
        if self.rank == 0:
            self.logger.info(f"Initializing MPIKMeans with {self.size} processes")
    
    def fit(self, X: np.ndarray) -> 'MPIKMeans':
        """
        Fir the K-means clustering model using MPI for parallel processing

        Args:
            X (np.ndarray): Input data matrix of shape (n_samples, n_features)

        Returns:
            self: The fitted model
        """
        try:
            if self.rank == 0:
                self.logger.info("Starting K-means clustering")
                
                # Ensure data is in the correct format
                X = np.asarray(X, dtype=np.float64)
                
                # Initialize centroids on root process
                if self.rank == 0:
                    np.random.seed(self.random_state)
                    idx = np.random.choice(len(X), self.n_clusters, replace=False)
                    centroids = X[idx]
                    self.logger.info("Initialized centroids")
                else:
                    centroids = None
                
                # Broadcast initial centroids to all process
                centroids = self.comm.bcast(centroids, root=0)
                
                # Split data among processes
                local_chunk_size = len(X) // self.size
                start_idx = self.rank * local_chunk_size
                end_idx = start_idx + local_chunk_size if self.rank != self.size - 1 else len(X)
                local_X = X[start_idx:end_idx]
                
                self.logger.debug(f"Process {self.rank} received {len(local_X)} samples")
                
                for iteration in range(self.max_iter):
                    old_centroids = centroids.copy()
                    
                    # Calculate local assignments
                    distances = np.sqrt(((local_X - centroids[:, np.newaxis])**2).sum(axis=2))
                    local_labels = distances.argmin(axis=0)
                    
                    # Update centroids
                    new_centroids = np.zeros_like(centroids)
                    counts = np.zeros(self.n_clusters)
                    
                    for i in range(self.n_clusters):
                        mask = local_labels == i
                        if mask.any():
                            new_centroids[i] = local_X[mask].sum(axis=0)
                            counts[i] = mask.sum()
                    
                    # Reduce results across all processes
                    centroids = self.comm.allreduce(new_centroids, op=MPI.SUM)
                    total_counts = self.comm.allreduce(counts, op=MPI.SUM)
                    
                    # Normalize centroids
                    for i in range(self.n_clusters):
                        if total_counts[i] > 0:
                            centroids[i] /= total_counts[i]
                    
                    # Check convergence
                    if np.all(np.abs(old_centroids - centroids) < self.tol):
                        if self.rank == 0:
                            self.logger.info(f"Converged after { iteration + 1 } iterations")
                            break
                    
                    if self.rank == 0 and (iteration + 1) % 10 == 0:
                        self.logger.info(f"Completed iteration { iteration + 1 }")
                        
                self.centroids_ = centroids
                return self
        except Exception as e:
            self.logger.error(f"Error in fit method: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster for each sample in X.

        Args:
            X (np.ndarray): Input data matrix

        Returns:
            np.ndarray: Cluster labels for each sample
        """
        try:
            X = np.asarray(X, dtype=np.float64)
            distances = np.sqrt(((X - self.centroids_[:, np.newaxis])**2).sum(axis=2))
            return distances.argmin(axis=0)
        except Exception as e:
            self.logger.error(f"Error in predict method: {str(e)}")
            raise