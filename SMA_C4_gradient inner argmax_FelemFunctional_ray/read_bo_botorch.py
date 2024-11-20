from glob import glob
import os
import random
from joblib import load
import numpy as np

grad_path = 'results\\results_budget_80_20241118\\*pkl'

def traj_proc(blend_path):
    data = []
    for p in glob(blend_path):
        _x, _p = load(p)
        # _rd_p = _p[:27]
        # random.shuffle(_rd_p)
        # _p = _rd_p + _p[27:]
        data.append(np.maximum.accumulate(_p))

    data = np.array(data)
    data = np.concatenate((
        np.vstack((np.arange(len(data.T)), data.mean(axis = 0), data.std(axis = 0))).T,
        data.T
    ), axis = -1)

    return data

    # rand_seed_header = [-3, -2, -1] + [int(p.split('-')[2]) for p in glob(blend_path)]

    # return np.vstack((rand_seed_header, data))

save_path = os.path.join(os.path.dirname(grad_path), '~res_sep.txt')
np.savetxt(save_path, traj_proc(grad_path), delimiter = '\t')