import numpy as np
from typing import List
import torch

NP_NON_ZERO_EPS = 1e-5

def weight_avg_func(comp: List[float], feature: List[float]) -> float:
    if not isinstance(comp, np.ndarray):
        comp = np.array(comp)
    if not isinstance(feature, np.ndarray):
        feature = np.array(feature)
    
    idx = np.where(comp > NP_NON_ZERO_EPS)
    comp, feature = comp[idx], feature[idx]
    return np.dot(comp, feature)

def delta_func(comp: List[float], feature: List[float]) -> float:
    idx = np.where(comp > NP_NON_ZERO_EPS)
    comp, feature = comp[idx], feature[idx]
    _weight_avg = weight_avg_func(comp, feature)
    return np.sqrt(weight_avg_func([(1 - f / _weight_avg) ** 2 for f in feature], comp))

def max_func(comp: List[float], feature: List[float]) -> float:
    idx = np.where(comp > NP_NON_ZERO_EPS)
    return np.max(feature[idx])

def min_func(comp: List[float], feature: List[float]) -> float:
    idx = np.where(comp > NP_NON_ZERO_EPS)
    return np.min(feature[idx])

def range_func(comp: List[float], feature: List[float]) -> float:
    return max_func(comp, feature) - min_func(comp, feature)

def maxc_func(comp: List[float], feature: List[float]) -> float:
    idx = np.where(comp > NP_NON_ZERO_EPS)
    comp, feature = comp[idx], feature[idx]
    idx = np.argmax(feature)
    return comp[idx] * feature[idx] / weight_avg_func(comp, feature)

def minc_func(comp: List[float], feature: List[float]) -> float:
    idx = np.where(comp > NP_NON_ZERO_EPS)
    comp, feature = comp[idx], feature[idx]
    idx = np.argmin(feature)
    return comp[idx] * feature[idx] / weight_avg_func(comp, feature)

def rangec_func(comp: List[float], feature: List[float]) -> float:
    idx = np.where(comp > NP_NON_ZERO_EPS)
    comp, feature = comp[idx], feature[idx]
    idx_min, idx_max = np.argmin(feature), np.argmax(feature)
    return comp[idx_min] * comp[idx_max] * (feature[idx_max] - feature[idx_min])


'''
    Tensor version of feature calculation functions.
    Define feature calculation functions for gradient optimization.
    Now feature calculation is considered as a functional.
    The input is composition, the output is calculated materials feature.
    Up-level the functions to ensure compatibility with job.Parallel.
'''
TORCH_NON_ZERO_EPS = 1e-5
SIGMOID_TEMP = 1e6
SOFTMAX_TEMP = 1e6

def soft_argmax(a: torch.Tensor) -> torch.Tensor:
    eps_len = len(a)
    eps_sequence = torch.linspace(1e-4, 0, eps_len)
    a = a + eps_sequence
    weights = torch.softmax(a * SOFTMAX_TEMP, dim=0)
    return weights

def weight_avg_func_torch(comp: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    mask = comp > TORCH_NON_ZERO_EPS
    return (comp * mask * feature).sum()

def delta_func_torch(comp: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    mask = comp > TORCH_NON_ZERO_EPS
    comp_masked = comp[mask]
    feature_masked = feature[mask]
    
    _weight_avg = (comp_masked * feature_masked).sum()
    
    # new feature values
    _new_feat_val = ((1 - feature_masked / _weight_avg) ** 2)
    
    return torch.sqrt((comp_masked * _new_feat_val).sum())

def max_func_torch(comp: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    filter_mask = comp > TORCH_NON_ZERO_EPS
    feature = feature[filter_mask]
    comp = comp[filter_mask]
    mask = torch.sigmoid((comp - TORCH_NON_ZERO_EPS) * SIGMOID_TEMP)
    argmax_weights = soft_argmax(feature)
    return (feature * argmax_weights * mask).sum()

def min_func_torch(comp: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    filter_mask = comp > TORCH_NON_ZERO_EPS
    feature = feature[filter_mask]
    comp = comp[filter_mask]
    mask = torch.sigmoid((comp - TORCH_NON_ZERO_EPS) * SIGMOID_TEMP)
    argmin_weights = soft_argmax(- feature)
    return (feature * argmin_weights * mask).sum()

def range_func_torch(comp: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    filter_mask = comp > TORCH_NON_ZERO_EPS
    feature = feature[filter_mask]
    comp = comp[filter_mask]
    mask = torch.sigmoid((comp - TORCH_NON_ZERO_EPS) * SIGMOID_TEMP)
    argmax_weights = soft_argmax(feature)
    argmin_weights = soft_argmax(- feature)
    return (feature * argmax_weights * mask).sum() - (feature * argmin_weights * mask).sum()

def maxc_func_torch(comp: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    mask = comp > TORCH_NON_ZERO_EPS
    comp = comp[mask]
    feature = feature[mask]
    weights = soft_argmax(feature)
    return (comp * weights * feature).sum() / (comp * feature).sum()

def minc_func_torch(comp: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    mask = comp > TORCH_NON_ZERO_EPS
    comp = comp[mask]
    feature = feature[mask]
    weights = soft_argmax(- feature)
    return (comp * weights * feature).sum() / (comp * feature).sum()

def rangec_func_torch(comp: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    mask = comp > TORCH_NON_ZERO_EPS
    comp = comp[mask]
    feature = feature[mask]
    
    # weights for soft argmax
    max_weights = soft_argmax(feature)
    
    # weights for soft argmin
    min_weights = soft_argmax(- feature)
    
    return (comp * max_weights).sum() \
           * (comp * min_weights).sum() \
           * (
                (feature * max_weights).sum() 
                - (feature * min_weights).sum()
            )