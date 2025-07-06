"""
Dimensionality Reduction Module

This module provides simple implementations of PCA and UMAP for reducing
the dimensionality of ClinicalBERT embeddings.

Note: t-SNE has been removed due to fundamental issues with transforming 
new data without data leakage, making it unsuitable for ML evaluation.
"""

import numpy as np
from sklearn.decomposition import PCA
import umap
from typing import Tuple, Optional
from loguru import logger
import os

# Get number of jobs from environment variable or default to 1
# Cap at 32 for UMAP to prevent threading issues
N_JOBS = min(int(os.environ.get('SKLEARN_N_JOBS', 1)), 32)

class DimensionalityReducer:
    """Base class for dimensionality reduction techniques."""
    
    def __init__(self, n_components: int, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.reducer = None
        
    def fit(self, X: np.ndarray):
        """Fit the reducer on the training data."""
        # For most reducers, we can use fit_transform and discard the result
        self.fit_transform(X)
        return self
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the reducer and transform the data."""
        raise NotImplementedError
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using the fitted reducer."""
        if self.reducer is None:
            raise ValueError("Reducer must be fitted first")
        return self.reducer.transform(X)


class PCAReducer(DimensionalityReducer):
    """PCA dimensionality reduction."""
    
    def __init__(self, n_components: int, random_state: int = 42):
        super().__init__(n_components, random_state)
        self.reducer = PCA(n_components=n_components, random_state=random_state)
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        logger.info(f"Applying PCA to reduce from {X.shape[1]} to {self.n_components} dimensions")
        return self.reducer.fit_transform(X)


class UMAPReducer(DimensionalityReducer):
    """UMAP dimensionality reduction."""
    
    def __init__(self, n_components: int, random_state: int = 42, n_neighbors: int = 15):
        super().__init__(n_components, random_state)
        self.n_neighbors = n_neighbors
        
        # Cap n_jobs for UMAP to prevent segfaults
        umap_n_jobs = min(N_JOBS, 16)  # UMAP works best with fewer threads
        
        self.reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            n_jobs=umap_n_jobs,
            low_memory=True,  # Better memory efficiency for large datasets
            verbose=False,    # Reduce output spam
            n_epochs=200,     # Reduce from default 500 for faster computation
            learning_rate=1.0 # Default learning rate
        )
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        # Adjust n_neighbors based on sample size
        max_neighbors = X.shape[0] - 1
        if self.n_neighbors >= max_neighbors:
            self.n_neighbors = max(2, max_neighbors // 2)
            
            # Recreate reducer with adjusted parameters
            umap_n_jobs = min(N_JOBS, 16)
            self.reducer = umap.UMAP(
                n_components=self.n_components,
                random_state=self.random_state,
                n_neighbors=self.n_neighbors,
                min_dist=0.1,
                n_jobs=umap_n_jobs,
                low_memory=True,
                verbose=False,
                n_epochs=200,
                learning_rate=1.0
            )
            
        logger.info(f"Applying UMAP to reduce from {X.shape[1]} to {self.n_components} dimensions")
        logger.info(f"Using n_neighbors: {self.n_neighbors}, n_jobs: {min(N_JOBS, 16)}")
        return self.reducer.fit_transform(X)


def get_reducer(method: str, n_components: int, random_state: int = 42) -> DimensionalityReducer:
    """Factory function to get the appropriate reducer."""
    method = method.lower()
    
    if method == 'pca':
        return PCAReducer(n_components, random_state)
    elif method == 'umap':
        return UMAPReducer(n_components, random_state)
    else:
        raise ValueError(f"Unknown reduction method: {method}. Supported methods: 'pca', 'umap'") 