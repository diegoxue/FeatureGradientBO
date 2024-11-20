from copy import deepcopy
import random
from typing import Callable, Tuple
import uuid
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

from sklearn import preprocessing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import RegressorMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, RBF, WhiteKernel, Matern, ExpSineSquared, RationalQuadratic
)

from joblib import Parallel, delayed

from model_env import N_ELEM, N_ELEM_FEAT, N_ELEM_FEAT_P1, N_PROC
from model_env import CnnDnnModel
from feature_selection_ga import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机数种子
set_seed(0) # default 0

seeds = np.random.randint(0, 9999, (9999, ))

def load_data():
    # Load the default dataset
    data = pd.read_excel('data\DSC-DATA-0613.xlsx', skiprows=[1,])
    
    # composition labels 
    comp_labels = ['Ti', 'Ni', 'Cu', 'Hf', 'Co', 'Zr', 'Fe', 'Pd', 'Ta', 'Nb', 'V']
    
    # processing condition labels 
    proc_labels = ['冷轧变形量', '退火处理温度', '退火处理时间']

    # property labels, one by one for ['enthalpy（Heating）', 'Ms', 'Mp', 'Mf', 'As', 'Ap', 'Af']
    prop_labels = ['Mp']
    print(f'loading SMA data ...')

    ''' delete proc condition all 0 items '''
    _mask = (data[proc_labels[1:]] == 0).all(axis = 1)
    data = data[~ _mask]
    print(f'deleted {sum(_mask)} items')

    ''' delete non usual proc conditions '''
    _mask = (data[proc_labels[0]] == 0.) & (data[proc_labels[1]] == 1273.) & (data[proc_labels[2]] == 1)
    data = data[_mask]
    print(f'deleted {sum(~ _mask)} items')
    print(f'remains {sum(_mask)} items')

    comp_data = data[comp_labels].to_numpy()
    proc_data = data[proc_labels].to_numpy()
    prop_data = data[prop_labels].to_numpy().ravel()

    elem_feature = pd.read_excel('data\\sma_element_features.xlsx')
    elem_feature = elem_feature[comp_labels].to_numpy()  # transpose: column for each elemental feature, row for each element 

    # (num_samples, num_elements), (num_samples, num_proc), (num_samples, num_prop), (num_elements, num_elem_features,)
    return comp_data, proc_data, prop_data, elem_feature

def cv_mae(x, 
           y, 
           model: GaussianProcessRegressor, 
           split_n, 
           kf_random_state: int    # TODO: no fixed random state
           ) -> float:
    ''' 
        Custom 10-fold cross validation.
    '''
    assert len(x) == len(y)

    # thread local variables
    x = deepcopy(x)
    y = deepcopy(y)
    model = clone(model)

    scaler = preprocessing.RobustScaler()
    x = scaler.fit_transform(x)
    kf = KFold(split_n, shuffle = True, random_state = kf_random_state)

    y_test_buff, y_pred_buff = [], []
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_test_buff += y_test.tolist()
        y_pred_buff += y_pred.tolist()

    return mean_absolute_error(y_test_buff, y_pred_buff)

def par_cv_mae(x, 
               y,  
               random_seed_list: List[int],
               n_split = 10,
               ) -> float:
    return  Parallel(n_jobs = -1)(delayed(cv_mae)(x, 
                                                  y, 
                                                  get_gpr(),
                                                  n_split, 
                                                  rs, ) for rs in random_seed_list)

if __name__ == '__main__':
    pass

    ''' 1. 准备数据 '''
    comp_data, _, prop_data, elem_feature = load_data()
    wavg_feature = np.dot(comp_data, elem_feature.T)

    comp_res_buff, feat_res_buff = [], []
    for _ in range(64):
        ''' 2. 随机选取100个成分-性能数据 '''
        selected_index = np.random.choice(len(comp_data), 100, replace = False)
        sel_comp_data = comp_data[selected_index]
        sel_wavg_feature = wavg_feature[selected_index]
        sel_prop_data = prop_data[selected_index]

        ''' 3. 使用GA选取最合适的4个weighted avg feature '''
        env = Env(sel_wavg_feature, sel_prop_data)
        ga = FeatureSelectionGA(env, verbose = 1)
        ga.generate(n_pop = 50, cxpb = 0.8, mutxpb = 0.1, ngen = 50)
        sel_f_idx_list = ga.get_fitest_ind().f_idx_list

        ''' 4. 分别使用comp及atom feat作为输入x评估模型的mae, 比如64次 '''
        comp_to_eval = sel_comp_data
        wavg_to_eval = (sel_wavg_feature.T[sel_f_idx_list]).T

        random_seeds_list = [np.random.randint(0, 999) for _ in range(64)]
        comp_res_buff.append(par_cv_mae(comp_to_eval, sel_prop_data, random_seeds_list))
        feat_res_buff.append(par_cv_mae(wavg_to_eval, sel_prop_data, random_seeds_list))

        joblib.dump(
            (comp_res_buff[-1], feat_res_buff[-1]),
            f'feat_eff_res-{str(uuid.uuid4())[:8]}.pkl'
        )
    
    joblib.dump(
        (comp_res_buff, feat_res_buff),
        'feat_eff_res.pkl'
    )