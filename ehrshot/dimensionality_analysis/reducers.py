"""
Dimensionality Reduction Module

This module provides simple implementations of PCA, t-SNE, and UMAP for reducing
the dimensionality of ClinicalBERT embeddings.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from typing import Tuple, Optional
from loguru import logger


class DimensionalityReducer:
    """Base class for dimensionality reduction techniques."""
    
    def __init__(self, n_components: int, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.reducer = None
        
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


class TSNEReducer(DimensionalityReducer):
    """t-SNE dimensionality reduction."""
    
    def __init__(self, n_components: int, random_state: int = 42, perplexity: float = 30.0):
        super().__init__(n_components, random_state)
        self.perplexity = min(perplexity, (X.shape[0] - 1) // 3) if hasattr(self, 'X') else perplexity
        self.reducer = TSNE(
            n_components=n_components, 
            random_state=random_state,
            perplexity=self.perplexity,
            n_iter=300,
            init='pca'
        )
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        # Adjust perplexity based on sample size
        max_perplexity = (X.shape[0] - 1) // 3
        if self.perplexity > max_perplexity:
            self.perplexity = max(5, max_perplexity)
            self.reducer = TSNE(
                n_components=self.n_components,
                random_state=self.random_state,
                perplexity=self.perplexity,
                n_iter=300,
                init='pca'
            )
        
        logger.info(f"Applying t-SNE to reduce from {X.shape[1]} to {self.n_components} dimensions")
        logger.info(f"Using perplexity: {self.perplexity}")
        return self.reducer.fit_transform(X)
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        # t-SNE cannot transform new data, need to refit
        logger.warning("t-SNE cannot transform new data, refitting on combined data")
        return self.fit_transform(X)


class UMAPReducer(DimensionalityReducer):
    """UMAP dimensionality reduction."""
    
    def __init__(self, n_components: int, random_state: int = 42, n_neighbors: int = 15):
        super().__init__(n_components, random_state)
        self.n_neighbors = n_neighbors
        self.reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=n_neighbors,
            min_dist=0.1
        )
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        # Adjust n_neighbors based on sample size
        max_neighbors = X.shape[0] - 1
        if self.n_neighbors >= max_neighbors:
            self.n_neighbors = max(2, max_neighbors // 2)
            self.reducer = umap.UMAP(
                n_components=self.n_components,
                random_state=self.random_state,
                n_neighbors=self.n_neighbors,
                min_dist=0.1
            )
            
        logger.info(f"Applying UMAP to reduce from {X.shape[1]} to {self.n_components} dimensions")
        logger.info(f"Using n_neighbors: {self.n_neighbors}")
        return self.reducer.fit_transform(X)


def get_reducer(method: str, n_components: int, random_state: int = 42) -> DimensionalityReducer:
    """Factory function to get the appropriate reducer."""
    method = method.lower()
    
    if method == 'pca':
        return PCAReducer(n_components, random_state)
    elif method == 'tsne':
        return TSNEReducer(n_components, random_state)
    elif method == 'umap':
        return UMAPReducer(n_components, random_state)
    else:
        raise ValueError(f"Unknown reduction method: {method}") 