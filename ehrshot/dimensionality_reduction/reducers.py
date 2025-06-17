"""
Dimensionality Reduction Techniques

This module implements various dimensionality reduction techniques optimized
for clinical embeddings, particularly focusing on improving k-NN performance
by addressing the curse of dimensionality.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import warnings

from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis
from sklearn.feature_selection import (
    VarianceThreshold, 
    SelectKBest, 
    SelectPercentile,
    f_classif, 
    mutual_info_classif
)
from sklearn.preprocessing import StandardScaler
from loguru import logger


class DimensionalityReducer(ABC):
    """Abstract base class for dimensionality reduction techniques."""
    
    def __init__(self, n_components: int, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.is_fitted = False
        self.scaler = None
        self.reducer = None
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'DimensionalityReducer':
        """Fit the reducer to the data."""
        pass
        
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data using the fitted reducer."""
        pass
        
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit the reducer and transform the data."""
        return self.fit(X, y).transform(X)
        
    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Get explained variance ratio if available."""
        if hasattr(self.reducer, 'explained_variance_ratio_'):
            return self.reducer.explained_variance_ratio_
        return None
        
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores if available."""
        if hasattr(self.reducer, 'scores_'):
            return self.reducer.scores_
        return None


class PCAReducer(DimensionalityReducer):
    """Principal Component Analysis reducer."""
    
    def __init__(self, n_components: int, random_state: int = 42, whiten: bool = False):
        super().__init__(n_components, random_state)
        self.whiten = whiten
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'PCAReducer':
        """Fit PCA to the data."""
        logger.info(f"Fitting PCA with {self.n_components} components")
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA
        self.reducer = PCA(
            n_components=self.n_components,
            random_state=self.random_state,
            whiten=self.whiten
        )
        self.reducer.fit(X_scaled)
        
        self.is_fitted = True
        logger.info(f"PCA fitted. Explained variance ratio: {self.reducer.explained_variance_ratio_[:5]}")
        logger.info(f"Total explained variance: {np.sum(self.reducer.explained_variance_ratio_):.3f}")
        
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted PCA."""
        if not self.is_fitted:
            raise ValueError("Reducer must be fitted before transform")
            
        X_scaled = self.scaler.transform(X)
        return self.reducer.transform(X_scaled)


class TruncatedSVDReducer(DimensionalityReducer):
    """Truncated SVD reducer (good for sparse matrices)."""
    
    def __init__(self, n_components: int, random_state: int = 42):
        super().__init__(n_components, random_state)
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'TruncatedSVDReducer':
        """Fit Truncated SVD to the data."""
        logger.info(f"Fitting Truncated SVD with {self.n_components} components")
        
        # Note: TruncatedSVD doesn't require centering, so we can optionally skip scaling
        # But for consistency with other methods, we'll still scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.reducer = TruncatedSVD(
            n_components=self.n_components,
            random_state=self.random_state
        )
        self.reducer.fit(X_scaled)
        
        self.is_fitted = True
        logger.info(f"Truncated SVD fitted. Explained variance ratio: {self.reducer.explained_variance_ratio_[:5]}")
        logger.info(f"Total explained variance: {np.sum(self.reducer.explained_variance_ratio_):.3f}")
        
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted Truncated SVD."""
        if not self.is_fitted:
            raise ValueError("Reducer must be fitted before transform")
            
        X_scaled = self.scaler.transform(X)
        return self.reducer.transform(X_scaled)


class FactorAnalysisReducer(DimensionalityReducer):
    """Factor Analysis reducer."""
    
    def __init__(self, n_components: int, random_state: int = 42, max_iter: int = 1000):
        super().__init__(n_components, random_state)
        self.max_iter = max_iter
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'FactorAnalysisReducer':
        """Fit Factor Analysis to the data."""
        logger.info(f"Fitting Factor Analysis with {self.n_components} components")
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.reducer = FactorAnalysis(
                n_components=self.n_components,
                random_state=self.random_state,
                max_iter=self.max_iter
            )
            self.reducer.fit(X_scaled)
        
        self.is_fitted = True
        logger.info("Factor Analysis fitted successfully")
        
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted Factor Analysis."""
        if not self.is_fitted:
            raise ValueError("Reducer must be fitted before transform")
            
        X_scaled = self.scaler.transform(X)
        return self.reducer.transform(X_scaled)


class VarianceThresholdReducer(DimensionalityReducer):
    """Remove features with low variance."""
    
    def __init__(self, threshold: float = 0.01):
        # Note: n_components is not used for this reducer
        super().__init__(n_components=None)
        self.threshold = threshold
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'VarianceThresholdReducer':
        """Fit variance threshold selector."""
        logger.info(f"Fitting Variance Threshold with threshold {self.threshold}")
        
        self.reducer = VarianceThreshold(threshold=self.threshold)
        self.reducer.fit(X)
        
        n_selected = self.reducer.get_support().sum()
        self.n_components = n_selected  # Update actual number of components
        
        self.is_fitted = True
        logger.info(f"Variance Threshold fitted. Selected {n_selected} features from {X.shape[1]}")
        
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted variance threshold."""
        if not self.is_fitted:
            raise ValueError("Reducer must be fitted before transform")
            
        return self.reducer.transform(X)


class UnivariateFeatureSelectionReducer(DimensionalityReducer):
    """Univariate feature selection based on statistical tests."""
    
    def __init__(self, n_components: int, score_func: str = 'f_classif', random_state: int = 42):
        super().__init__(n_components, random_state)
        
        # Choose scoring function
        if score_func == 'f_classif':
            self.score_func = f_classif
        elif score_func == 'mutual_info_classif':
            self.score_func = lambda X, y: mutual_info_classif(X, y, random_state=random_state)
        else:
            raise ValueError(f"Unknown score function: {score_func}")
            
        self.score_func_name = score_func
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'UnivariateFeatureSelectionReducer':
        """Fit univariate feature selector."""
        if y is None:
            raise ValueError("UnivariateFeatureSelectionReducer requires labels (y) for fitting")
            
        logger.info(f"Fitting Univariate Feature Selection with {self.n_components} features using {self.score_func_name}")
        
        self.reducer = SelectKBest(score_func=self.score_func, k=self.n_components)
        self.reducer.fit(X, y)
        
        self.is_fitted = True
        logger.info(f"Univariate Feature Selection fitted. Selected {self.n_components} features from {X.shape[1]}")
        
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted feature selector."""
        if not self.is_fitted:
            raise ValueError("Reducer must be fitted before transform")
            
        return self.reducer.transform(X)


def create_reducer(method: str, n_components: int, **kwargs) -> DimensionalityReducer:
    """Factory function to create dimensionality reducers.
    
    Args:
        method: Reduction method ('pca', 'truncated_svd', 'factor_analysis', 
                'variance_threshold', 'univariate_selection')
        n_components: Number of components to keep
        **kwargs: Additional parameters for the reducer
        
    Returns:
        DimensionalityReducer instance
    """
    reducers = {
        'pca': PCAReducer,
        'truncated_svd': TruncatedSVDReducer,
        'factor_analysis': FactorAnalysisReducer,
        'variance_threshold': VarianceThresholdReducer,
        'univariate_selection': UnivariateFeatureSelectionReducer,
    }
    
    if method not in reducers:
        raise ValueError(f"Unknown reduction method: {method}. Available: {list(reducers.keys())}")
        
    return reducers[method](n_components=n_components, **kwargs)


def get_optimal_n_components_pca(X: np.ndarray, 
                                 variance_threshold: float = 0.95,
                                 max_components: Optional[int] = None) -> int:
    """Find optimal number of PCA components to retain given variance threshold.
    
    Args:
        X: Input data
        variance_threshold: Minimum cumulative variance to retain
        max_components: Maximum number of components to consider
        
    Returns:
        Optimal number of components
    """
    if max_components is None:
        max_components = min(X.shape[0], X.shape[1]) - 1
        
    # Fit PCA with maximum components
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=max_components)
    pca.fit(X_scaled)
    
    # Find number of components for desired variance
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum_var >= variance_threshold) + 1
    
    logger.info(f"Optimal PCA components: {n_components} (explains {cumsum_var[n_components-1]:.3f} variance)")
    
    return n_components 