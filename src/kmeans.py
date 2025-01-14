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
        self.batch_size = 1000
        
        if self.rank == 0:
            self.logger.info(f"Initializing MPIKMeans with {self.size} processes")
            
    def _process_batch(self, batch_X, centroids):
        """Process a mini-batch of data"""
        distances = np.sqrt(((batch_X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = distances.argmin(axis=0)
        
        new_centroid = np.zeros_like(centroids)
        counts = np.zeros(self.n_clusters, dtype=np.int64)
        
        for i in range(self.n_clusters):
            mask = labels == i
            if mask.any():
                new_centroid[i] = batch_X[mask].sum(axis=0)
                counts[i] = mask.sum()
        
        return new_centroid, counts, labels

    def fit(self, X):
        try:
            X = np.asarray(X, dtype=np.float64)
            n_samples = len(X)
            
            # Initialize centroids
            if self.rank == 0:
                np.random.seed(self.random_state)
                indices = np.random.choice(n_samples, self.n_clusters, replace=False)
                centroids = X[indices]
            else:
                centroids = None
            
            centroids = self.comm.bcast(centroids, root=0)
            
            # Split data among processes
            local_size = n_samples // self.size
            start_idx = self.rank * local_size
            end_idx = start_idx + local_size if self.rank != self.size - 1 else n_samples
            local_X = X[start_idx:end_idx]
            
            for iteration in range(self.max_iter):
                old_centroids = centroids.copy()
                
                # Process in mini-batches
                n_batches = len(local_X) // self.batch_size + 1
                local_new_centroids = np.zeros_like(centroids)
                local_counts = np.zeros(self.n_clusters, dtype=np.int64)
                
                for i in range(n_batches):
                    start = i * self.batch_size
                    end = min(start + self.batch_size, len(local_X))
                    batch_X = local_X[start:end]
                    
                    if len(batch_X) == 0:
                        continue
                        
                    batch_centroids, batch_counts, _ = self._process_batch(batch_X, centroids)
                    local_new_centroids += batch_centroids
                    local_counts += batch_counts
                
                # Reduce results across processes
                global_centroids = self.comm.allreduce(local_new_centroids, op=MPI.SUM)
                global_counts = self.comm.allreduce(local_counts, op=MPI.SUM)
                
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
            predictions = []
            
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i + self.batch_size]
                distances = np.sqrt(((batch_X - self.centroids_[:, np.newaxis])**2).sum(axis=2))
                batch_predictions = distances.argmin(axis=0)
                predictions.append(batch_predictions)
                
            return np.concatenate(predictions)
        
        except Exception as e:
            self.logger.error(f"Error in predict method: {str(e)}")
            raise