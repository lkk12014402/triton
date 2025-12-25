"""
CUDA-to-SYCL Optimization Agent

This module provides tools for detecting CUDA kernel optimizations and
generating SYCL optimization recommendations.
"""

from .detector import ChangeDetector
from .analyzer import OptimizationAnalyzer
from .recommender import SyclRecommender
from .pr_generator import PRGenerator

__all__ = [
    'ChangeDetector',
    'OptimizationAnalyzer',
    'SyclRecommender',
    'PRGenerator',
]

__version__ = '1.0.0'
