from environment import *
from scipy.spatial.distance import pdist, squareform
from glob import glob
import numpy as np
import joblib
from sklearn.manifold import TSNE

grad_path = 'results\\tsne\\gradopt-45-exp-40-100-inner-80-q-1.pkl'
enum_path = 'results\\tsne\\enumeration-43f9557f-45-40-100.pkl'

def decode_key(key: str):
        return np.array(list(map(float, key.split('*'))))

def key(state: List[float]):
    return '*'.join(map(str, np.array(state).round(COMPOSITION_ROUNDUP_DIGITS)))

all_state_key_set = set()

''' lookup dict '''
state_key_2_tsne_cord = {}
state_key_2_prop = {}

def update(traj, prop):
     for _comp, _prop in zip(traj, prop):
          _key = key(_comp)
          if _key not in all_state_key_set:
               all_state_key_set.add(_key)
               state_key_2_prop[_key] = _prop

grad_traj, grad_prop = joblib.load(grad_path)
enum_traj, enum_prop = joblib.load(enum_path)

update(grad_traj, grad_prop)
update(enum_traj, enum_prop)

all_state_key_list = sorted(list(all_state_key_set))
all_state_repr_list = [decode_key(_s_key) for _s_key in all_state_key_list]

''' TSNE '''
tsne_cord_list = TSNE(n_components = 2, random_state = 33).fit_transform(
    np.array(all_state_repr_list)
).round(COMPOSITION_ROUNDUP_DIGITS)

# tsne cord lookup dict
for _key, _tsne_cord in zip(all_state_key_list, tsne_cord_list):
    state_key_2_tsne_cord[_key] = _tsne_cord

def tsne_process(comps: List[List[float]], props: List[float], dest_file_name: str):
    at_least2d = lambda x: np.array(x).reshape(-1, 1)

    _tsne_cords = np.array([state_key_2_tsne_cord[key(_x)] for _x in comps])
    _iter_no = np.hstack((np.zeros(40), np.arange(start = 1, stop = 100 - 40 + 1))).reshape(-1, 1)
    _res = np.concatenate((_tsne_cords, at_least2d(props), _iter_no, ), axis = 1)
    # sorted_indices = np.argsort(_res[:, -1])
    # _res = _res[sorted_indices]
    np.savetxt(
        dest_file_name,
        _res,
        delimiter = '\t'
    )

tsne_process(grad_traj, grad_prop, 'results\\tsne\\grad_45_tsne.txt')
tsne_process(enum_traj, enum_prop, 'results\\tsne\\enum_45_tsne.txt')