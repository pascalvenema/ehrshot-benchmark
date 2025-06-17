"""
Dimensionality Reduction Module for EHRSHOT Benchmark

This module provides various dimensionality reduction techniques
to analyze the impact of reduced embeddings on prediction performance,
particularly for k-NN models suffering from the curse of dimensionality.
"""

from .reducers import (
    PCAReducer,
    TruncatedSVDReducer,
    FactorAnalysisReducer,
    VarianceThresholdReducer,
    UnivariateFeatureSelectionReducer,
    create_reducer
)
from .utils import (
    load_features,
    save_reduced_features,
    get_optimal_dimensions,
    plot_explained_variance,
    create_reduction_report
)
from .evaluator import DimensionalityReductionEvaluator

__all__ = [
    'PCAReducer',
    'TruncatedSVDReducer', 
    'FactorAnalysisReducer',
    'VarianceThresholdReducer',
    'UnivariateFeatureSelectionReducer',
    'create_reducer',
    'load_features',
    'save_reduced_features',
    'get_optimal_dimensions',
    'plot_explained_variance',
    'create_reduction_report',
    'DimensionalityReductionEvaluator'
] 