import joblib
import glob
import numpy as np

paths = glob.glob('results\\20240721_BO_init100\\*.pkl')

bsf_buffer = []

for p in paths:
    _d = joblib.load(p)
    ''' previous impl. '''
    # _bsf = []
    # previous_len = 0
    # for _, _n_visited, _best in _d:
    #     _bsf.extend([_best] * (_n_visited - previous_len))
    #     previous_len = _n_visited
    # bsf_buffer.append(_bsf)
    ''' current impl. '''
    bsf_buffer.append(_d)

# min_len = min([len(_b) for _b in bsf_buffer])
# for i in range(len(bsf_buffer)): bsf_buffer[i] = bsf_buffer[i][:min_len]

max_len = max([len(_b) for _b in bsf_buffer])
for i in range(len(bsf_buffer)): bsf_buffer[i] = bsf_buffer[i] + [bsf_buffer[i][-1]] * (max_len - len(bsf_buffer[i]))

bsf_buffer = np.array(bsf_buffer).T

idx_col = np.arange(len(bsf_buffer)).reshape(-1, 1)
mean_col = bsf_buffer.mean(axis=-1).reshape(-1, 1)
std_col = bsf_buffer.std(axis=-1).reshape(-1, 1)

# bsf_buffer = np.concatenate((idx_col, mean_col, std_col, bsf_buffer), axis = -1).round(5)
bsf_buffer = np.concatenate((idx_col, mean_col, std_col), axis = -1).round(5)

np.savetxt(
    'bo_bsf_buffer.txt',
    bsf_buffer,
    delimiter='\t'
)
