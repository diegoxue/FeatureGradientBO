''' 
    TODO Description to add ...
'''
import functools
import warnings
import itertools
import math
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
import uuid
import warnings
import joblib
from loguru import logger
import numpy as np

import pandas as pd
import torch
from torch.distributions import Normal
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll_torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.sampling import manual_seed
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from botorch.optim import optimize_acqf

from scipy.stats import norm

from environment import (
    COMPOSITION_ROUNDUP_DIGITS, Environment, State, 
    COMP_LIMITS, ELEM_N, COMPOSITION_INTERVAL, EPISODE_LEN, COMP_MAX_LIMITS, COMP_MIN_LIMITS,
    get_ground_truth_func, get_mo_ground_truth_func, os_path
)
from feature_selection_ga import Env, FeatureSelectionGA

def ignore_all_warnings(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 忽略所有警告
            return func(*args, **kwargs)
    return wrapper

def ensure_rng(random_state = None):
    """
        Creates a random number generator based on an optional seed.  This can be
        an integer or another random state for a seeded rng, or None for an
        unseeded rng.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state

class AbstractCompositionBayesianOptimization(ABC):
    def __init__(self,
                 elem_num: int = 4,
                 init_rand_num: int = 40,
                 total_exp_num: int = 100,
                 init_seed: int  = 0,
                 ):
        self._ground_truth_func = get_mo_ground_truth_func()
        self._gpr = None
        self._gpr_noise_level = 1e-6

        self._elem_num = elem_num
        self._bounds = np.array(COMP_LIMITS[:elem_num])     # TODO: round
        
        self._init_rand_num = init_rand_num
        self._total_exp_num = total_exp_num

        self._all_elem_names = ['Ti', 'Mo', 'V', 'Ta', 'Nb', 'Cr', 'Mn']

        _all_elem_features = pd.read_excel(os_path('data\\elemental_features.xlsx'))
        self._elem_features = _all_elem_features[self._all_elem_names[:elem_num]].to_numpy()

        self._sel_elem_feat_idxs = None
        self._sel_elem_features = None
        self._sel_elem_feat_num = 4                         # TODO: make it modifiable

        self._acqf_func = self.ei_acqf
        self._norm_dist = Normal(0, 1)

        self._container = set()
        self._exp_comp_history = list()
        self._exp_target_history = list()

        self._init_seed = init_seed
        self.init_random_compositions(init_rand_num)
        # self.init_orthogonal_compositions()

    def init_random_compositions(self, num: int):
        ''' Initialize `num` random compositions '''
        _rand_comps = self.rejection_sample_compositions(num = num, seed = self._init_seed)
        for _comp in _rand_comps:
            _targ = self._ground_truth_func(_comp)
            self.register(_comp, _targ)

    def rejection_sample_compositions(self, 
                                      num: int, 
                                      tolerance = 1e-6, 
                                      seed = None):
        '''
            Generate `num` random initial compositions using rejection sampling.
        '''
        random_state = ensure_rng(seed)

        ''' pre-allocated legal actions for each elements '''
        buffered_actions = []
        for i in range(self._elem_num):
            _min, _max = COMP_MIN_LIMITS[i], COMP_MAX_LIMITS[i]
            _actions = np.linspace(_min, _max, round(((_max - _min) / COMPOSITION_INTERVAL)) + 1)\
                .round(COMPOSITION_ROUNDUP_DIGITS)
            buffered_actions.append(_actions)
        
        _buff_set = set()           # collision detection
        _comp_buff = []             # buffer to store sampled compositions
        while len(_comp_buff) < num:
            _comp = np.zeros(self._elem_num)
            _flag = True
            for i in range(self._elem_num - 1):
                _c = random_state.choice(buffered_actions[i])
                if _comp.sum() + _c > 1.0 + tolerance:
                    _flag = False   # early stop
                    break
                _comp[i] = _c
            _last_c = round(1 - _comp.sum(), COMPOSITION_ROUNDUP_DIGITS)
            if _last_c < COMP_MIN_LIMITS[-1] or _last_c > COMP_MAX_LIMITS[-1]:
                _flag = False       # early stop
            else:
                _comp[-1] = _last_c
            if _flag:
                _key = self.encode(_comp)
                if _key in _buff_set:
                    continue        # early stop
                _buff_set.add(_key)
                _comp_buff.append(self.round(_comp))
        
        return _comp_buff
    
    def feature_selection(self):
        ''' Select 4 elemental features using self-implemented genetic algorithm. '''
        _comps = np.array(self._exp_comp_history)
        _props = np.array(self._exp_target_history)

        # GA main loop
        _wavg_feature = np.dot(_comps, self._elem_features.T)
        env = Env(_wavg_feature, _props)
        ga = FeatureSelectionGA(env, verbose = 1)
        ''' TODO: maybe better hyper-parameters '''
        ga.generate(n_pop = 48, cxpb = 0.8, mutxpb = 0.1, ngen = 50)

        # buffer GA results
        self._sel_elem_feat_idxs = ga.get_fitest_ind().f_idx_list
        self._sel_elem_features = self._elem_features[self._sel_elem_feat_idxs].T

    def round(self, comp: List[float]) -> List[float]:
        if not isinstance(comp, np.ndarray):
            comp = np.array(comp)
        return comp.round(COMPOSITION_ROUNDUP_DIGITS)

    def encode(self, comp: List[float]) -> str:
        return '*'.join(map(str, self.round(comp)))

    def contains(self, comp: List[float]) -> bool:
        ''' If a composition is already experimented. '''
        key = self.encode(comp)
        return key in self._container

    def register(self, comp: List[float], target: float):
        ''' Register the experimented point into exp history buffer '''
        comp = self.round(comp)
        assert not self.contains(comp), self.encode(comp)
        key = self.encode(comp)
        self._container.add(key)
        self._exp_comp_history.append(comp)
        self._exp_target_history.append(target)
    
    @staticmethod
    def init_gpr():
        pass

    @abstractmethod
    def suggest_next_x(self) -> List[float]:
        pass

    @abstractmethod
    def cal_elemental_feature(self, comp: List[float]) -> List[float]:
        pass

    def run(self):
        _local_id = str(uuid.uuid4())[:8]

        ''' 1. identify important elemental feature subset '''
        self.feature_selection()

        ''' 2. BO outer loop '''
        for exp_round in range(self.exp_round + 1, self._total_exp_num + 1):
            _cal_feature = self.cal_elemental_feature(self._exp_comp_history)
            _target = np.array(self._exp_target_history)

            self.fit(_cal_feature, _target)
            
            ''' 3. BO inner loop '''
            _next_comp = self.suggest_next_x()
            _next_target = self._ground_truth_func(_next_comp)

            self.register(_next_comp, _next_target)

            ''' 4. log '''
            logger.info(f'{_local_id} - round {exp_round}, best-so-far {self.best_so_far}')

    def update_elem_feat_bounds(self, elem_feat_arr: np.ndarray):   # TODO: check data-type
        _feat_min_bounds = elem_feat_arr.min(axis = 0)
        _feat_max_bounds = elem_feat_arr.max(axis = 0)
        
        _zero_mask = (_feat_min_bounds == _feat_max_bounds)
        _feat_max_bounds[_zero_mask] = _feat_max_bounds[_zero_mask] + 1
        
        self._elem_feat_bounds = np.vstack((_feat_min_bounds, _feat_max_bounds))

    # TODO: tensorize
    def standardize(self, elem_feat_arr: np.ndarray) -> List[float]:
        ''' Apply min-max standardization and store the bounds for current exp.ed points '''
        return (elem_feat_arr - self._elem_feat_bounds[0]) / \
            (self._elem_feat_bounds[1] - self._elem_feat_bounds[0])

    @ignore_all_warnings
    def fit(self, 
            elem_feat: np.ndarray, 
            target: np.ndarray):
        '''
            Fit a botorch GPR model.

            Parameters
            ----------
            elem_feat: numpy array
                unnormalized elemental feature 
            target: numpy array
                target property 
        '''
        assert len(elem_feat) == len(target)                        # TODO: remove this in production code

        self.update_elem_feat_bounds(elem_feat)
        standardized_elem_feat = self.standardize(elem_feat)
        self._y_max = max(target)

        # tensorize
        if isinstance(standardized_elem_feat, np.ndarray):
            train_x = torch.tensor(standardized_elem_feat)
        else:
            train_x = standardized_elem_feat
        train_y = torch.tensor(target).reshape(-1, 1)
        train_Yvar = torch.full_like(train_y, self._gpr_noise_level)    # fixed low-level noise

        self._gpr = SingleTaskGP(train_x, train_y, train_Yvar)
        mll = ExactMarginalLogLikelihood(self._gpr.likelihood, self._gpr)
        fit_gpytorch_mll_torch(mll)

    # TODO: tensorize and maintain tensor graph
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ''' 
            GPR prediction.
            Parameters
            ----------
            elem_feat: numpy array
                unseen unnormalized elemental feature 
        '''
        x = self.standardize(x.reshape(-1, self._sel_elem_feat_num))
        x = torch.tensor(x)
        y_posterior = self._gpr.posterior(x)
        y_mean = y_posterior.mean.detach().numpy()
        y_std = y_posterior.variance.detach().sqrt().numpy()    # std
        return y_mean.flatten(), y_std.flatten()
    
    # TODO debug
    def ei_acqf(self, mean, std, xi = 0.0):
        ''' Calculate expected improvement. '''
        if isinstance(mean, np.ndarray):
            ei_vals = np.zeros(shape = mean.shape)
            _cdf_func = lambda x: norm.cdf(x)
            _pdf_func = lambda x: norm.pdf(x)
        elif isinstance(mean, torch.Tensor):
            ei_vals = torch.zeros_like(mean)
            _cdf_func = lambda x: self._norm_dist.cdf(x)
            _pdf_func = lambda x: self._norm_dist.log_prob(x).exp()

        mask = std > 1e-6
        mean, std = mean[mask], std[mask]
        a = (mean - self._y_max - xi)
        z = a / std
        ei_vals[mask] = a * _cdf_func(z) + std * _pdf_func(z)
        return ei_vals
    
    def save(self):
        joblib.dump(
            (self._exp_comp_history, self._exp_target_history),
            f'{str(uuid.uuid4())[:8]}-{self._init_seed}-{self._init_rand_num}-{self._total_exp_num}.pkl'
        )

    @property
    def exp_round(self):
        return len(self._exp_comp_history)

    @property
    def best_so_far(self):
        return max(self._exp_target_history)
