from joblib import load
from glob import glob
import numpy as np

enum_path = 'results\\SMA_CO_enum&gradOpt_no_diff\\enumeration\\*pkl'
grad_path = 'results\\SMA_CO_enum&gradOpt_no_diff\\gradOpt\\*pkl'

def traj_proc(blend_path):
    data = []
    for p in glob(blend_path):
        _x, _p = load(p)
        data.append(np.maximum.accumulate(_p))

    data = np.array(data)
    data = np.concatenate((
        np.vstack((np.arange(len(data.T)), data.mean(axis = 0), data.std(axis = 0))).T,
        data.T
    ), axis = -1)

    rand_seed_header = [-3, -2, -1] + [int(p.split('-')[1]) for p in glob(blend_path)]

    return np.vstack((rand_seed_header, data))

np.savetxt('enum_res.txt', traj_proc(enum_path), delimiter = '\t')
np.savetxt('grad_res.txt', traj_proc(grad_path), delimiter = '\t')
    