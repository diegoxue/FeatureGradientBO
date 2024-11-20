from glob import glob
import os
from joblib import load
import numpy as np

grad_path = 'results\\results_ti_c4_grad_20240908\\*pkl'

def traj_proc(blend_path):
    data = []
    for p in glob(blend_path):
        _x, _p = load(p)
        data.append(np.maximum.accumulate(_p))

        # for comp, prop in zip(_x, _p):
        #     print(comp, prop)
        # exit()

    data = np.array(data)
    data = np.concatenate((
        np.vstack((np.arange(len(data.T)), data.mean(axis = 0), data.std(axis = 0))).T,
        data.T
    ), axis = -1)

    rand_seed_header = [-3, -2, -1] + [int(p.split('-')[2]) for p in glob(blend_path)]

    return np.vstack((rand_seed_header, data))

trajs = traj_proc(grad_path)
save_path = os.path.join(os.path.dirname(grad_path), '~res.txt')
np.savetxt(save_path, trajs, delimiter = '\t')
np.savetxt(save_path.replace('~res', '~res_no_std'), trajs[1:][:, :3], delimiter = '\t')