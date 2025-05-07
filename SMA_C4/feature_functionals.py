"""
Feature Functional Module for Materials Informatics

This module provides various feature transformation functions to transform materials composition into materials features.
These functions implement a wide range of descriptors used in materials informatics to transform
elemental compositions and properties into meaningful features for surrogate models in Bayesian optimization.

Both NumPy and PyTorch implementations are provided.
"""

import numpy as np
from typing import List
import torch

# Small epsilon value to avoid numerical issues with zero-valued compositions
NP_NON_ZERO_EPS = 1e-5

def weight_avg_func(comp: List[float], feature: List[float]) -> float:
    """
    Calculate weighted average of elemental features based on composition fractions
    
    This is one of the most fundamental materials descriptors,
    modeling a property as the composition-weighted average of constituent elements' properties.
    
    Args:
        comp: List of composition fractions summing to 1.0
        feature: List of corresponding elemental property values
    
    Returns:
        Weighted average value
    """
    if not isinstance(comp, np.ndarray):
        comp = np.array(comp)
    if not isinstance(feature, np.ndarray):
        feature = np.array(feature)
    
    # Filter out elements with negligible content
    idx = np.where(comp > NP_NON_ZERO_EPS)
    comp, feature = comp[idx], feature[idx]
    return np.dot(comp, feature)

def delta_func(comp: List[float], feature: List[float]) -> float:
    """
    Calculate composition-weighted property deviation (delta parameter)
    
    This parameter quantifies, for example, the atomic size mismatch or property fluctuation
    and is important in multi-component alloy design to predict, for example, solid solution formation.
    
    Args:
        comp: List of composition fractions summing to 1.0
        feature: List of corresponding elemental property values
    
    Returns:
        Delta parameter value
    """
    idx = np.where(comp > NP_NON_ZERO_EPS)
    comp, feature = comp[idx], feature[idx]
    _weight_avg = weight_avg_func(comp, feature)
    # Calculate root mean square deviation from weighted average
    return np.sqrt(weight_avg_func([(1 - f / _weight_avg) ** 2 for f in feature], comp))

def max_func(comp: List[float], feature: List[float]) -> float:
    """
    Find maximum value of a property among constituent elements
    
    Useful for capturing limiting properties where the maximum value
    dominates the material behavior (e.g., maximum atomic radius).
    
    Args:
        comp: List of composition fractions summing to 1.0
        feature: List of corresponding elemental property values
    
    Returns:
        Maximum property value
    """
    idx = np.where(comp > NP_NON_ZERO_EPS)
    return np.max(feature[idx])

def min_func(comp: List[float], feature: List[float]) -> float:
    """
    Find minimum value of a property among constituent elements
    
    Useful for capturing limiting properties where the minimum value
    dominates the material behavior (e.g., minimum melting temperature).
    
    Args:
        comp: List of composition fractions summing to 1.0
        feature: List of corresponding elemental property values
    
    Returns:
        Minimum property value
    """
    idx = np.where(comp > NP_NON_ZERO_EPS)
    return np.min(feature[idx])

def range_func(comp: List[float], feature: List[float]) -> float:
    """
    Calculate range of property values (max - min)
    
    Quantifies the diversity or spread of an elemental property within the alloy,
    which can correlate with, for example, phase stability and solid solution strengthening.
    
    Args:
        comp: List of composition fractions summing to 1.0
        feature: List of corresponding elemental property values
    
    Returns:
        Range of property values
    """
    return max_func(comp, feature) - min_func(comp, feature)

def maxc_func(comp: List[float], feature: List[float]) -> float:
    """
    Calculate composition-weighted maximum property contribution
    
    This descriptor captures both the magnitude of a maximum property value
    and the composition fraction of the element that possesses it.
    
    Args:
        comp: List of composition fractions summing to 1.0
        feature: List of corresponding elemental property values
    
    Returns:
        Weighted maximum property contribution
    """
    idx = np.where(comp > NP_NON_ZERO_EPS)
    comp, feature = comp[idx], feature[idx]
    idx = np.argmax(feature)
    return comp[idx] * feature[idx] / weight_avg_func(comp, feature)

def minc_func(comp: List[float], feature: List[float]) -> float:
    """
    Calculate composition-weighted minimum property contribution
    
    This descriptor captures both the magnitude of a minimum property value
    and the composition fraction of the element that possesses it.
    
    Args:
        comp: List of composition fractions summing to 1.0
        feature: List of corresponding elemental property values
    
    Returns:
        Weighted minimum property contribution
    """
    idx = np.where(comp > NP_NON_ZERO_EPS)
    comp, feature = comp[idx], feature[idx]
    idx = np.argmin(feature)
    return comp[idx] * feature[idx] / weight_avg_func(comp, feature)

def rangec_func(comp: List[float], feature: List[float]) -> float:
    """
    Calculate composition-weighted property range contribution
    
    Combines the composition fractions of elements with extreme property values
    and the magnitude of their difference.
    
    Args:
        comp: List of composition fractions summing to 1.0
        feature: List of corresponding elemental property values
    
    Returns:
        Weighted property range contribution
    """
    idx = np.where(comp > NP_NON_ZERO_EPS)
    comp, feature = comp[idx], feature[idx]
    idx_min, idx_max = np.argmin(feature), np.argmax(feature)
    return comp[idx_min] * comp[idx_max] * (feature[idx_max] - feature[idx_min])


"""
PyTorch Tensor Implementations of Feature Functions

These are differentiable versions of the above functions for gradient-based optimization.
By implementing these functions with PyTorch tensors and operations, we enable automatic
differentiation through the entire feature calculation pipeline, which is essential for
gradient-based Bayesian optimization of material compositions.
"""
TORCH_NON_ZERO_EPS = 1e-5
SIGMOID_TEMP = 1e6  # Temperature parameter for sigmoid function (controls sharpness)
SOFTMAX_TEMP = 1e6  # Temperature parameter for softmax function (controls sharpness)

def soft_argmax(a: torch.Tensor) -> torch.Tensor:
    """
    Differentiable approximation of argmax using softmax
    
    This function provides a smooth, differentiable approximation to the non-differentiable
    argmax operation, enabling gradient flow through operations that would normally
    involve discrete selection.
    
    Args:
        a: Input tensor
        
    Returns:
        Softmax weights that approximate one-hot encoding of argmax
    """
    eps_len = len(a)
    eps_sequence = torch.linspace(1e-4, 0, eps_len)
    a = a + eps_sequence
    weights = torch.softmax(a * SOFTMAX_TEMP, dim=0)
    return weights

def weight_avg_func_torch(comp: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    """
    Differentiable implementation of weighted average function
    
    Args:
        comp: Tensor of composition fractions
        feature: Tensor of elemental property values
        
    Returns:
        Weighted average as a scalar tensor
    """
    mask = comp > TORCH_NON_ZERO_EPS
    return (comp * mask * feature).sum()

def delta_func_torch(comp: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    """
    Differentiable implementation of delta parameter calculation
    
    Args:
        comp: Tensor of composition fractions
        feature: Tensor of elemental property values
        
    Returns:
        Delta parameter as a scalar tensor
    """
    mask = comp > TORCH_NON_ZERO_EPS
    comp_masked = comp[mask]
    feature_masked = feature[mask]
    
    _weight_avg = (comp_masked * feature_masked).sum()
    
    # Calculate squared deviations from weighted average
    _new_feat_val = ((1 - feature_masked / _weight_avg) ** 2)
    
    return torch.sqrt((comp_masked * _new_feat_val).sum())

def max_func_torch(comp: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    """
    Differentiable approximation of maximum function
    
    Uses soft argmax to create a differentiable version of the max operation.
    
    Args:
        comp: Tensor of composition fractions
        feature: Tensor of elemental property values
        
    Returns:
        Approximate maximum value as a scalar tensor
    """
    filter_mask = comp > TORCH_NON_ZERO_EPS
    feature = feature[filter_mask]
    comp = comp[filter_mask]
    mask = torch.sigmoid((comp - TORCH_NON_ZERO_EPS) * SIGMOID_TEMP)
    argmax_weights = soft_argmax(feature)
    return (feature * argmax_weights * mask).sum()

def min_func_torch(comp: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    """
    Differentiable approximation of minimum function
    
    Uses soft argmax on negative values to create a differentiable version of the min operation.
    
    Args:
        comp: Tensor of composition fractions
        feature: Tensor of elemental property values
        
    Returns:
        Approximate minimum value as a scalar tensor
    """
    filter_mask = comp > TORCH_NON_ZERO_EPS
    feature = feature[filter_mask]
    comp = comp[filter_mask]
    mask = torch.sigmoid((comp - TORCH_NON_ZERO_EPS) * SIGMOID_TEMP)
    argmin_weights = soft_argmax(- feature)  # Negative to find minimum
    return (feature * argmin_weights * mask).sum()

def range_func_torch(comp: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    """
    Differentiable approximation of range function
    
    Combines differentiable max and min functions to calculate the range.
    
    Args:
        comp: Tensor of composition fractions
        feature: Tensor of elemental property values
        
    Returns:
        Approximate range value as a scalar tensor
    """
    filter_mask = comp > TORCH_NON_ZERO_EPS
    feature = feature[filter_mask]
    comp = comp[filter_mask]
    mask = torch.sigmoid((comp - TORCH_NON_ZERO_EPS) * SIGMOID_TEMP)
    argmax_weights = soft_argmax(feature)
    argmin_weights = soft_argmax(- feature)
    return (feature * argmax_weights * mask).sum() - (feature * argmin_weights * mask).sum()

def maxc_func_torch(comp: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation of maxc function
    
    Differentiable weighted maximum contribution calculation.
    
    Args:
        comp: Tensor of composition fractions
        feature: Tensor of elemental property values
        
    Returns:
        Weighted maximum contribution as a scalar tensor
    """
    mask = comp > TORCH_NON_ZERO_EPS
    comp = comp[mask]
    feature = feature[mask]
    weights = soft_argmax(feature)
    return (comp * weights * feature).sum() / (comp * feature).sum()

def minc_func_torch(comp: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation of minc function
    
    Differentiable weighted minimum contribution calculation.
    
    Args:
        comp: Tensor of composition fractions
        feature: Tensor of elemental property values
        
    Returns:
        Weighted minimum contribution as a scalar tensor
    """
    mask = comp > TORCH_NON_ZERO_EPS
    comp = comp[mask]
    feature = feature[mask]
    weights = soft_argmax(- feature)
    return (comp * weights * feature).sum() / (comp * feature).sum()

def rangec_func_torch(comp: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation of rangec function
    
    Differentiable weighted range contribution calculation.
    
    Args:
        comp: Tensor of composition fractions
        feature: Tensor of elemental property values
        
    Returns:
        Weighted range contribution as a scalar tensor
    """
    mask = comp > TORCH_NON_ZERO_EPS
    comp = comp[mask]
    feature = feature[mask]
    
    # Weights for soft argmax and argmin
    max_weights = soft_argmax(feature)
    min_weights = soft_argmax(- feature)
    
    return (comp * max_weights).sum() \
           * (comp * min_weights).sum() \
           * (
                (feature * max_weights).sum() 
                - (feature * min_weights).sum()
            )