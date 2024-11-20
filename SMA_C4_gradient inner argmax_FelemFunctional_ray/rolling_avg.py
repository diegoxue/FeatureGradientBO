import numpy as np

def rolling_average(data, window_size = 20):
    averages = []
    _idx_sub = window_size // 2
    _idx_pls = window_size - _idx_sub
    for i in range(len(data)):
        _idx_start = max(0, i - _idx_sub)
        _idx_end = min(len(data), i + _idx_pls)
        window = data[_idx_start : _idx_end]
        average = np.mean(window)
        averages.append(average)
    return averages

path = 'mp_validate_log.txt'
data = np.loadtxt(path)

_func = lambda x : np.array(rolling_average(x)).reshape(-1, 1)

data = np.concatenate(
    (data, _func(data[:, 1]), _func(data[:, 2]), _func(data[:, 3])),
    axis=-1
)

np.savetxt(
    path.replace('.txt', '_rlavg.txt'),
    data,
    delimiter='\t'
)