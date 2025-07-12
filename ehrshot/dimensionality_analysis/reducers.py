
import numpy as np
from sklearn.decomposition import PCA
import umap
from typing import Tuple, Optional
from loguru import logger
import os


N_JOBS = min(int(os.environ.get('SKLEARN_N_JOBS', 1)), 32)

class DimensionalityReducer:
    
    def __init__(self, n_components: int, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.reducer = None
        
    def fit(self, X: np.ndarray):
        self.fit_transform(X)
        return self
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def transform(self, X: np.ndarray) -> np.ndarray:

        return self.reducer.transform(X)


class PCAReducer(DimensionalityReducer):
    
    def __init__(self, n_components: int, random_state: int = 42):
        super().__init__(n_components, random_state)
        self.reducer = PCA(n_components=n_components, random_state=random_state)
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.reducer.fit_transform(X)


class UMAPReducer(DimensionalityReducer):
    
    def __init__(self, n_components: int, random_state: int = 42, n_neighbors: int = 15):
        super().__init__(n_components, random_state)
        self.n_neighbors = n_neighbors
        
        umap_n_jobs = min(N_JOBS, 16)  
        
        self.reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            n_jobs=umap_n_jobs,
            low_memory=True,  
            verbose=False,    
            n_epochs=200,     
            learning_rate=1.0 
        )
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        max_neighbors = X.shape[0] - 1
        if self.n_neighbors >= max_neighbors:
            self.n_neighbors = max(2, max_neighbors // 2)
            
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
            
        return self.reducer.fit_transform(X)


def get_reducer(method: str, n_components: int, random_state: int = 42) -> DimensionalityReducer:
    method = method.lower()
    
    if method == 'pca':
        return PCAReducer(n_components, random_state)
    elif method == 'umap':
        return UMAPReducer(n_components, random_state)
