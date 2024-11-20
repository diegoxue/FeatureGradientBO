import joblib
import numpy as np

path = 'bayes_opt-sma-discrete-max_seed-12-240719-b31b858d.pkl'

data = joblib.load(path)

data = list(zip(*data))[1]

data = np.array(data)
data_mean = data.mean(axis = 0).reshape(-1, 1)
data_std = data.std(axis = 0).reshape(-1, 1)
data_idx = np.linspace(0, len(data_mean)-1, len(data_mean)).reshape(-1, 1)

np.savetxt(
    path.replace('pkl', 'txt'),
    np.concatenate((data_idx, data_mean, data_std), axis = 1),
    delimiter='\t'
)