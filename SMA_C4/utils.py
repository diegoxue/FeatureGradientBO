"""
    This file contains black box test function for Bayesian optimization.

    NOT USED IN THIS PROJECT.
"""

import random
import joblib
import torch
import numpy as np
from scipy.interpolate import Rbf

''' func to bias x coordinates '''
# NOTE done in later codes

''' func to domain warp x coordinates '''
FUNC_MAX_DIM = 10

def ackley(x, a = 20, b = 0.2, c = 2 * np.pi):
    """
        x: vector of input values
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    d = len(x) # dimension of input vector x
    sum_sq_term = -a * np.exp(-b * np.sqrt(sum(x * x) / d))
    cos_term = -np.exp(sum(np.cos(c * x) / d))
    return a + np.exp(1) + sum_sq_term + cos_term

def rastrigin(x, a = 10):
    """
        x: vector of input values
    """
    d = len(x)
    return a * d + sum([xi**2 - a * np.cos(2 * np.pi * xi) for xi in x])

def levy(x):
    """
        x: vector of input values
    """
    d = len(x)
    func_sum = 0.
    for i in range(d):
        w = 1 + (x[i] - 1) / 4
        if i == 0:
            _tmp = np.sin(np.pi * w) ** 2
        elif i == d - 1:
            _tmp = (w - 1) ** 2 * (1 + 10 * np.sin(np.pi * w + 1) ** 2)
        else:
            _tmp = (w - 1) ** 2 * (1 + np.sin(np.pi * 2 * w) ** 2)
        func_sum += _tmp
    return func_sum

def griewank(x, a = 4000, b = 1):
    """
        x: vector of input values
    """
    d = len(x)
    sum_sq_term = sum([xi**2 for xi in x]) / a
    prod_cos_term = np.prod([np.cos(xi / np.sqrt(i + 1)) for i, xi in enumerate(x)])
    griewank_val = sum_sq_term - prod_cos_term + b
    # return np.log10(griewank_val + 1e-10)
    return griewank_val

def dejong(x):
    """
        x: vector of input values
    """
    d = len(x)
    return sum([xi**2 for xi in x])

def save(args, save_name, model, ep):
    import os
    save_dir = './trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")

func_params = {
    'ackley': {
        'func': ackley,
        'x_min': -5,
        'x_max': 5,
        'act_dim': 51,
    },
    'rastrigin': {
        'func': rastrigin,
        'x_min': -10,
        'x_max': 10,
        'act_dim': 2001,
    },
    'levy': {
        'func': levy,
        'x_min': -10,
        'x_max': 10,
        'act_dim': 51,
    },
    'griewank': {
        'func': griewank,
        'x_min': -200,
        'x_max': 200,
        'act_dim': 51,
    },
    'dejong': {
        'func': dejong,
        'x_min': -5,
        'x_max': 5,
        'act_dim': 51,
    },
}