import numpy as np
from mpi4py import MPI
import logging
from .utils import setup_logging

class MPIKMeans:
    def __init__(self, config, n_clusters):
        self.config = config
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.logger = setup_logging(f"MPIKMeans_Process_{self.rank}")
        
        # Initialize parameters
        self.n_clusters = n_clusters
        self.max_iter = config.max_iter
        self.tol = config.tolerance
        self.random_state = config.random_state
        
        if self.rank == 0:
            self.logger.info(f"Initializing MPIKMeans with {self.size} processes")

    def fit(self, X):
        try:
            if self.rank == 0:
                self.logger.info("Starting K-means clustering")
            
            # Ensure data is in correct format
            X = np.asarray(X, dtype=np.float64)
            
            # Initialize centroids on root process
            if self.rank == 0:
                np.random.seed(self.random_state)
                idx = np.random.choice(len(X), self.n_clusters, replace=False)
                centroids = X[idx]
                self.logger.info("Initialized centroids")
            else:
                centroids = None
            
            # Broadcast centroids to all processes
            centroids = self.comm.bcast(centroids, root=0)
            
            # Split data among processes
            chunk_size = len(X) // self.size
            start_idx = self.rank * chunk_size
            end_idx = start_idx + chunk_size if self.rank != self.size - 1 else len(X)
            local_X = X[start_idx:end_idx]
            
            self.logger.info(f"Process {self.rank} received {len(local_X)} samples")
            
            for iteration in range(self.max_iter):
                old_centroids = centroids.copy()
                
                # Calculate distances and assignments
                distances = np.sqrt(((local_X - centroids[:, np.newaxis])**2).sum(axis=2))
                local_labels = distances.argmin(axis=0)
                
                # Calculate new centroids
                new_centroids = np.zeros_like(centroids)
                counts = np.zeros(self.n_clusters, dtype=np.int64)
                
                for i in range(self.n_clusters):
                    mask = local_labels == i
                    if mask.any():
                        new_centroids[i] = local_X[mask].sum(axis=0)
                        counts[i] = mask.sum()
                
                # Reduce results across processes
                global_centroids = self.comm.allreduce(new_centroids, op=MPI.SUM)
                global_counts = self.comm.allreduce(counts, op=MPI.SUM)
                
                # Update centroids
                for i in range(self.n_clusters):
                    if global_counts[i] > 0:
                        centroids[i] = global_centroids[i] / global_counts[i]
                
                # Check convergence
                diff = np.max(np.abs(old_centroids - centroids))
                if self.rank == 0:
                    self.logger.info(f"Iteration {iteration + 1}, max change: {diff:.6f}")
                
                if diff < self.tol:
                    if self.rank == 0:
                        self.logger.info(f"Converged after {iteration + 1} iterations")
                    break
            
            self.centroids_ = centroids
            return self
            
        except Exception as e:
            self.logger.error(f"Error in fit method: {str(e)}")
            self.comm.Abort(1)
            raise

    def predict(self, X):
        try:
            X = np.asarray(X, dtype=np.float64)
            distances = np.sqrt(((X - self.centroids_[:, np.newaxis])**2).sum(axis=2))
            return distances.argmin(axis=0)
        except Exception as e:
            self.logger.error(f"Error in predict method: {str(e)}")
            raise