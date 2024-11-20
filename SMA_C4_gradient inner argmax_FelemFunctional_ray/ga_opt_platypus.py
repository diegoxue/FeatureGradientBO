import numpy as np
from platypus import NSGAII, Problem, Real, Integer

from environment import *

def nsgaii_traj():
    func = get_mo_ground_truth_func()
    comp_sum = round(1.0 / COMPOSITION_INTERVAL)

    candidate_set = set()
    traj_buffer = []
    num_func_calls = [0]
    tolerance = 0.01

    def to_maximize(vars):
        # global traj_buffer, num_func_calls, candidate_set
        x_0 = round(vars[0] / comp_sum, COMPOSITION_ROUNDUP_DIGITS)
        x_1 = round(vars[1] / comp_sum, COMPOSITION_ROUNDUP_DIGITS)
        x_2 = round(vars[2] / comp_sum, COMPOSITION_ROUNDUP_DIGITS)
        x_3 = round(vars[3] / comp_sum, COMPOSITION_ROUNDUP_DIGITS)
        x_4 = round(vars[4] / comp_sum, COMPOSITION_ROUNDUP_DIGITS)
        x_5 = round(vars[5] / comp_sum, COMPOSITION_ROUNDUP_DIGITS)
        x_6 = round(vars[6] / comp_sum, COMPOSITION_ROUNDUP_DIGITS)
        x_7 = round(vars[7] / comp_sum, COMPOSITION_ROUNDUP_DIGITS)
        x_8 = round(vars[8] / comp_sum, COMPOSITION_ROUNDUP_DIGITS)
        x_9 = round(vars[9] / comp_sum, COMPOSITION_ROUNDUP_DIGITS)
        x_10 = round(vars[10] / comp_sum, COMPOSITION_ROUNDUP_DIGITS)
        x = np.array([
            x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10
        ])
        _func_val = func(x / x.sum())
        num_func_calls.append(num_func_calls[-1] + 1)
        _key = '*'.join(map(str, x))
        if abs(x.sum() - 1.) <= tolerance and _key not in candidate_set:
            traj_buffer.append(_func_val)
            candidate_set.add(_key)
        return [_func_val], [abs(x.sum() - 1.)]

    problem = Problem(ELEM_N, 1, 1)
    problem.types[:] = [
        Integer(round(_min / COMPOSITION_INTERVAL), round(_max / COMPOSITION_INTERVAL)) \
        for _min, _max in zip(COMP_MIN_LIMITS, COMP_MAX_LIMITS)
    ]
    problem.directions[:] = Problem.MAXIMIZE
    problem.constraints[:] = f"<={tolerance}"
    problem.function = to_maximize

    num_of_runs = 15000
    algorithm = NSGAII(problem)
    algorithm.run(num_of_runs)

    return np.maximum.accumulate(traj_buffer[:1501])

# nsgaii_traj(), exit()
num_trajs = 24
trajs = joblib.Parallel(n_jobs = 8)(joblib.delayed(nsgaii_traj)() for sd in range(num_trajs))

traj_data = np.array(trajs)
traj_data_mean = traj_data.mean(axis = 0).reshape(-1, 1)
traj_data_std = traj_data.std(axis = 0).reshape(-1, 1)
traj_data_idxs = np.arange(len(traj_data_mean)).reshape(-1, 1)

np.savetxt(
    'ga_nsgaii.txt',
    np.concatenate((traj_data_idxs, traj_data_mean, traj_data_std), axis = -1),
    delimiter='\t'
)


