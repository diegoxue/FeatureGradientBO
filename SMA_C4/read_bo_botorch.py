"""
    This script reads the results of Bayesian optimization from a given path
    and processes them to generate a trajectory of BSF values, with mean and std.
"""

from glob import glob
import os
import random
from joblib import load
import numpy as np

grad_path = 'path_to_bo_results\\*pkl'

def traj_proc(blend_path):
    data = []
    for p in glob(blend_path):
        """
            _x: (num_samples, num_elements), 
            _p: (num_samples, ) : property values
        """
        _x, _p = load(p)
        data.append(np.maximum.accumulate(_p))

    data = np.array(data)
    data = np.concatenate((
        np.vstack((np.arange(len(data.T)), data.mean(axis = 0), data.std(axis = 0))).T,
        data.T
    ), axis = -1)

    return data

save_path = os.path.join(os.path.dirname(grad_path), '~res_sep.txt')
np.savetxt(save_path, traj_proc(grad_path), delimiter = '\t')