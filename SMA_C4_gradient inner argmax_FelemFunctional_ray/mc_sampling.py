from environment import *

num_trajs = 24

def mc_traj(max_ep_len = 1501,
            tolerance = 1e-6,
            ):
    
    buffered_actions = []
    for i in range(ELEM_N):
        _min, _max = COMP_MIN_LIMITS[i], COMP_MAX_LIMITS[i]
        _actions = np.linspace(_min, _max, round(((_max - _min) / COMPOSITION_INTERVAL)) + 1)\
            .round(COMPOSITION_ROUNDUP_DIGITS)
        buffered_actions.append(_actions)

    func = get_mo_ground_truth_func()

    _traj = []
    while len(_traj) < max_ep_len:
        _tmp = np.zeros(ELEM_N)
        _flag = True
        for i in range(ELEM_N - 1):
            _c = np.random.choice(buffered_actions[i])
            if _tmp.sum() + _c > 1.0 + tolerance:
                _flag = False
                break
            _tmp[i] = _c
        _last_c = round(1 - _tmp.sum(), COMPOSITION_ROUNDUP_DIGITS)
        if _last_c < COMP_MIN_LIMITS[-1] or _last_c > COMP_MAX_LIMITS[-1]:
            _flag = False
        else:
            _tmp[-1] = _last_c
        if _flag:
            _func_val = func(_tmp)
            _traj.append(_func_val if not _traj else max(_func_val, _traj[-1]))
            if len(_traj) % 100 == 0: print(f'no. {len(_traj)}')
        
    return _traj


trajs = joblib.Parallel(n_jobs = 8)(joblib.delayed(mc_traj)() for sd in range(num_trajs))

traj_data = np.array(trajs)
traj_data_mean = traj_data.mean(axis = 0).reshape(-1, 1)
traj_data_std = traj_data.std(axis = 0).reshape(-1, 1)
traj_data_idxs = np.arange(len(traj_data_mean)).reshape(-1, 1)

np.savetxt(
    'random_sampled_trajs.txt',
    np.concatenate((traj_data_idxs, traj_data_mean, traj_data_std), axis = -1),
    delimiter='\t'
)