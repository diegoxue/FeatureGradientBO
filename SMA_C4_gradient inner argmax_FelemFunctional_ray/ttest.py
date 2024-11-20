import glob
import joblib
import numpy as np
from scipy.stats import ttest_ind

# BO part
paths = glob.glob('results\\20240721_BO_init100\\*.pkl')

bsf_buffer = []

for p in paths:
    _d = joblib.load(p)
    bsf_buffer.append(_d[-1])

bo_data = bsf_buffer

print(bo_data)

# bo_data = np.array([0.625, 0.6095, 0.6243, 0.6483, 0.6201, 0.6256, 0.6069, 0.6092, 0.6241, 0.6155, 0.6063, 0.6191])

# print('bo stat.:', bo_data.mean(), bo_data.std(),)

# RL part
paths = glob.glob('results\\20240720_RL_direct_R_4\\*.pkl')

bsf_buffer = []

for p in paths:
    _d = joblib.load(p)
    bsf_buffer.append(_d)

for i in range(len(bsf_buffer)): bsf_buffer[i] = bsf_buffer[i][:1500]

bsf_buffer = np.array(bsf_buffer).T

rl_data = bsf_buffer[-1, :]

'''
    unpaired t-test
'''
# 执行配对t检验
# t_statistic, p_value = ttest_ind(bo_data, rl_data[:12])
t_statistic, p_value = ttest_ind(bo_data, rl_data)

# 输出t统计量和p值
print('t statistic: %.8f, p value: %.8f' % (t_statistic, p_value))