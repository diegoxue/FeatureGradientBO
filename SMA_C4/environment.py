'''
"""
This script serves as the ground truth environment for Bayesian Optimization (BO) in shape memory alloy (SMA) design.
Its main functions include:

1. Property Prediction:
   - Uses pre-trained neural network models to predict key SMA properties:
     * Enthalpy of transformation
     * Martensite peak temperature (Mp)
     * Austenite peak temperature (Ap)
   - These predictions serve to compose the objective function (figure of merit, FOM) for BO optimization

2. Composition Space Definition:
   - Defines valid composition ranges for alloying elements (Ti, Ni, Cu, Hf, etc.)

This environment acts as a "ground truth" for expensive physical experiments,
allowing benchmark of Bayesian Optimization (BO) algorithms.
"""
'''
from __future__ import annotations
from collections import namedtuple
import os
import math
import random
from typing import List
from copy import deepcopy
import uuid
import warnings
from bayes_opt import UtilityFunction
import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from bayes_opt.util import ensure_rng
import torch

from model_env_train import get_model


# Genaral data structure definition
Transition = namedtuple('Transition', ('current_state', 'action', 'delayed_reward', 'next_state'))
TrainingIndicator = namedtuple('TrainingIndicator', ('epoch', 'loss', 'total_q'))
CompositionLimit = namedtuple('CompositionLimit', ('min_bound', 'max_bound'))

OUTPUT_BLOCK_SIZE = 100

''' class: State '''
# state.py
ActionType = float
EPISODE_LEN = 10
REPLAY_MEMORY_PATH = 'replay_memory_buffer.pk'
MEMORY_CAPACITY = 3000
RESUME_MEMORY_BUFFER = False
COMPOSITION_INTERVAL = 0.001
COMPOSITION_ROUNDUP_DIGITS = 4

''' compositional bounds, NOTE USER PARAMETERS '''
TI_MIN, TI_MAX = 0.20, 0.55
NI_MIN, NI_MAX = 0.20, 0.55
CU_MIN, CU_MAX = 0., 0.20
HF_MIN, HF_MAX = 0., 0.20
CO_MIN, CO_MAX = 0., 0.1
ZR_MIN, ZR_MAX = 0., 0.20
FE_MIN, FE_MAX = 0., 0.1
PD_MIN, PD_MAX = 0., 0.33
TA_MIN, TA_MAX = 0., 0.05
NB_MIN, NB_MAX = 0., 0.20
V_MIN, V_MAX = 0., 0.1

""" composition limit ensemble """
COMP_LIMITS =(
    CompositionLimit(TI_MIN, TI_MAX),
    CompositionLimit(NI_MIN, NI_MAX),
    CompositionLimit(CU_MIN, CU_MAX),
    CompositionLimit(HF_MIN, HF_MAX),
    CompositionLimit(CO_MIN, CO_MAX),
    CompositionLimit(ZR_MIN, ZR_MAX),
    CompositionLimit(FE_MIN, FE_MAX),
    CompositionLimit(PD_MIN, PD_MAX),
    CompositionLimit(TA_MIN, TA_MAX),
    CompositionLimit(NB_MIN, NB_MAX),
    CompositionLimit(V_MIN, V_MAX),
)

COMP_MIN_LIMITS = CompositionLimit(*zip(*COMP_LIMITS)).min_bound
COMP_MAX_LIMITS = CompositionLimit(*zip(*COMP_LIMITS)).max_bound

COMP_MULTIPLIER = 100.          # sum=1 -> sum=100.

ELEM_TOTAL = 11
ELEM_N = len(COMP_LIMITS)

EPSILON_START = 0.9
EPSILON_DECAY_COEF = 10000
EPSILON_END = 0.1
LEARNING_RATE = 1e-3            # Modification needed!
RL_TRAINING_EPOCHS = 1000       # Modification needed!
DEFAULT_LOG_INTERVAL = 1000     # terminal log every this epochs
RL_SAMPLE_BATCH_SIZE = 128
GAMMA = 0.80                    # TODO original 0.8
TARGET_UPDATE_PERIOD = 10
DQL_AGENT_PATH = 'dql_agent.pt'
DQL_TRAINING_INDICATOR_PATH = 'rl_agent_training_indicators.pk'
# composition tuning limits
COMP_LOW_BOUND_INT = round(min(COMP_MIN_LIMITS) / COMPOSITION_INTERVAL)
COMP_HIGH_BOUND_INT = round(max(COMP_MAX_LIMITS) / COMPOSITION_INTERVAL)
COMP_LOW_BOUND = COMP_LOW_BOUND_INT * COMPOSITION_INTERVAL
COMP_HIGH_BOUND = COMP_HIGH_BOUND_INT * COMPOSITION_INTERVAL
# action definition
ALL_ACTIONS = [round(x * COMPOSITION_INTERVAL, COMPOSITION_ROUNDUP_DIGITS) \
                for x in range(COMP_LOW_BOUND_INT, COMP_HIGH_BOUND_INT + 1)]
ALL_ACTIONS_COUNT = len(ALL_ACTIONS)
ACTIONS_TO_INDEX_DICT = dict(zip(ALL_ACTIONS, range(ALL_ACTIONS_COUNT)))

ROUND_DIGIT = 3
STATE_DELIMETER_CHAR = '*'


# for path compatibility
def os_path(path: str):
    return path.replace('\\\\', '\\').replace('\\', os.sep).replace('/', os.sep)

def get_ground_truth_func(model_path = 'model\model.pth', data_path = 'model\data.pth'):
    '''
        Return the func that maps a composition -> a mechanical property (mp / en).
    '''
    model, d, scalers = get_model(os_path(model_path), os_path(data_path), resume = True)
    model.eval()    # NOTE important

    comp_scaler = scalers[0]
    proc_scaler = scalers[1]
    prop_scaler = scalers[2]

    # TODO modification needed, NOTE keep it fixed
    base_proc = np.array([0, 1273, 1], dtype=np.float32).reshape(1, -1)
    elem_ft = d[-1]

    # pre-calculated processing condition data
    _proc = proc_scaler.transform(base_proc)
    _proc = torch.tensor(_proc, dtype=torch.float32).reshape(1, 1, d[1].shape[-1], 1)

    def _func(x):
        ''' maps a composition (a list with the sumation of 1.) to its SMA property prediction '''
        x = (np.array(x) * COMP_MULTIPLIER).round(ROUND_DIGIT)
        # assert len(x) == ELEM_N, f'len(x) != {ELEM_N}: {len(x)}'
        _comp = x.reshape(1, -1)

        _comp = comp_scaler.transform(_comp)

        _comp = torch.tensor(_comp, dtype=torch.float32).reshape(1, 1, d[0].shape[-1], 1)
        elem_t = torch.tensor(elem_ft, dtype=torch.float32).reshape(1, 1, *(elem_ft.shape))

        _prop = model(_comp, elem_t, _proc).detach().numpy()
        _prop = prop_scaler.inverse_transform(_prop)

        return _prop.item()
    
    return _func

def get_mo_ground_truth_func():
    '''
    Build a ground truth function (figure of merit, FOM) for Bayesian optimization
    
    This function creates a composite objective function that combines multiple 
    property predictions (enthalpy, transformation temperatures) into a single scalar value.
    The combined objective prioritizes:
    1. Higher enthalpy (energy storage capacity)
    2. Narrower transformation hysteresis (difference between Ap and Mp)
    3. Transformation temperatures centered around a target temperature (300K)
    
    This serves as the "ground truth" for Bayesian optimization, mimicking
    experimental measurements that would be performed in real materials discovery.
    
    Returns:
        Callable function that takes composition as input and returns scalar objective value
    '''
    # Load individual property prediction models
    _en_func = get_ground_truth_func('model\\en_model.pth', 'model\\en_data.pth',)
    _mp_func = get_ground_truth_func('model\\mp_model.pth', 'model\\mp_data.pth',)
    _ap_func = get_ground_truth_func('model\\ap_model.pth', 'model\\ap_data.pth',)

    # Target transformation temperature (room temperature in K)
    _dest_T = 300

    # Normalization scales for objective components
    # These control the relative importance of each property in the combined objective
    _mo_scale = np.array([
        35.,    # Enthalpy scale (J/g)
        50.,    # Transformation hysteresis scale (K)
        50.,    # Temperature offset from target scale (K)
    ])

    def _mo_func(x):
        '''
        Multi-objective function for alloy composition evaluation
        
        This function evaluates a composition by predicting multiple properties
        and combining them into a single objective value to be maximized.
        
        Args:
            x: List of element compositions (atomic fractions) in the alloy
            
        Returns:
            Normalized scalar objective value (higher is better)
        '''
        # Pad composition with zeros if shorter than total elements
        if len(x) < ELEM_TOTAL:
            x = list(x) + [0.] * (ELEM_TOTAL - len(x))
            
        # Predict individual properties using surrogate models
        _en_val = _en_func(x)  # Enthalpy (energy storage capacity)
        _mp_val = _mp_func(x)  # Martensite peak temperature
        _ap_val = _ap_func(x)  # Austenite peak temperature

        # Calculate objective components:
        _mo_val = np.array([
            _en_val,  # 1. Enthalpy (higher is better, raw value)
            _mo_scale[1] - abs(_ap_val - _mp_val),  # 2. Transformation hysteresis (narrower is better)
            _mo_scale[2] - abs(_dest_T - (_mp_val + _ap_val) / 2)  # 3. Proximity to target temperature
        ])

        # Normalize and average the components to get final objective
        return (_mo_val / _mo_scale).mean()
    
    return _mo_func

"""
NOTE: We do not use the following class.
"""

class State:
    def __init__(self, if_init: bool = False, 
                 previous_state: State = None, 
                 action: ActionType = None, 
                 episode_len = EPISODE_LEN):
        if if_init:
            # atomic fraction (%) of [C, Al, V, Cr, Mn, Fe, Co, Ni, Cu, Mo]
            # self.__composition = [0., 0., 0.1, 0.15, 0., 0.4, 0.1, 0.25, 0., 0.,]   # TODO further tests needed
            self.__composition = [0.] * ELEM_N
            self.__episode_len = episode_len
            self.__episode_count = -1
        else:
            self.__composition = deepcopy(previous_state.get_composition())
            previous_episode_no = previous_state.get_episode_count()
            self.__episode_len = episode_len
            self.__episode_count = previous_episode_no + 1

            ''' 
                substitution rule:
                    2024.07.08 - Only reach a rational composition when the episode ends.
            '''
            sub_idx = self.__episode_count
            _min = max(COMP_MIN_LIMITS[sub_idx], 1 - sum(self.__composition[:sub_idx]) - sum(COMP_MAX_LIMITS[sub_idx + 1:]))
            _max = min(COMP_MAX_LIMITS[sub_idx], 1 - sum(self.__composition[:sub_idx]))
            _min, _max = round(_min, COMPOSITION_ROUNDUP_DIGITS), round(_max, COMPOSITION_ROUNDUP_DIGITS)
            assert _min <= action <= _max
            self.__composition[sub_idx] = action

            if sub_idx == self.__episode_len - 1:
                self.__composition[-1] = 1. - sum(self.__composition[:-1])

            # round up compositions
            for idx in range(len(self.__composition)):
                self.__composition[idx] = round(self.__composition[idx], \
                    COMPOSITION_ROUNDUP_DIGITS)

    def get_episode_len(self) -> int:
        return self.__episode_len

    def get_episode_count(self) -> int:
        return self.__episode_count
    
    def get_composition(self):
        return self.__composition

    def repr(self):
        # len(feature) corresponds to flattened dimensions in DqlModel.
        feature = deepcopy(self.__composition)
        feature.append(self.__episode_count)
        return feature

    def done(self):
        return self.__episode_count == self.__episode_len - 1

    def get_action_idx_limits(self):
        '''
            Get action limits according to current state.

            @output:    (composition_min_idx, composition_max_idx)
                            composition_min_idx * COMPOSITION_INTERVAL == composition_lower_limit_in_float
                            composition_max_idx * COMPOSITION_INTERVAL == composition_upper_limit_in_float
        '''
        elem_index = self.__episode_count + 1
        assert elem_index < self.__episode_len, f'elem_index: {elem_index}, ELEM_N: {ELEM_N}'
        _min = max(COMP_MIN_LIMITS[elem_index], 1 - sum(self.__composition[:elem_index]) - sum(COMP_MAX_LIMITS[elem_index + 1:]))
        _max = min(COMP_MAX_LIMITS[elem_index], 1 - sum(self.__composition[:elem_index]))

        comp_min_idx = ACTIONS_TO_INDEX_DICT[round(_min, COMPOSITION_ROUNDUP_DIGITS)]
        comp_max_idx = ACTIONS_TO_INDEX_DICT[round(_max, COMPOSITION_ROUNDUP_DIGITS)]
        return comp_min_idx, comp_max_idx

    def generate_random_action(self, random_seed = None) -> ActionType:
        '''
            Generate one random action that can be applied to this state

            @output:    a random action in float
        '''
        random_state = ensure_rng(random_seed)
        comp_min_idx, comp_max_idx = self.get_action_idx_limits()
        rand_comp_idx = random_state.randint(comp_min_idx, comp_max_idx + 1)
        return ALL_ACTIONS[rand_comp_idx]
        return rand_comp_idx
    
    @staticmethod
    def encode_key(x):
        return STATE_DELIMETER_CHAR.join(map(str, x))
    
    @staticmethod
    def decode_key(key: str):
        return np.array(list(map(float, key.split(STATE_DELIMETER_CHAR)))).round(ROUND_DIGIT)   # check return data type

class Environment:
    def __init__(self, 
                 init_N = 50, 
                 enable_ei: bool = False,
                 random_seed = None):
        self.init_world_model()

        # self.all_actions = np.linspace(self.x_min, self.x_max, self.act_dim).round(ROUND_DIGIT).tolist()
        self.state_dim = len(self.reset().repr())
        self.act_dim = ALL_ACTIONS_COUNT

        self.best_score = float('-inf')
        self.best_x = None

        '''
            "The answer to the ultimate question of life, 
            the universe, and everything is 42."
        '''
        self._random_state = 42
        self.surrogate = None
        self.surrogate_buffer = set()   # stores experimented xs
        self.surrogate_buffer_list = []
        self.cached_surrogate_pred_dict = dict()

        self.init_N = init_N
        self.enable_ei = enable_ei
        self.init_surrogate(self.init_N, random_seed)
    
    def init_world_model(self,):
        self.func = get_mo_ground_truth_func()

    def copy_initialization(self, env: Environment):
        ''' copy the randomly initialized exp point from a ref environment instance '''
        self.best_score = float('-inf')
        self.best_x = None

        self.surrogate_buffer = deepcopy(env.surrogate_buffer)
        self.surrogate_buffer_list = deepcopy(env.surrogate_buffer_list)

        # train the model using copied initial exp points
        self.update_surrogate()

    def init_surrogate(self, init_N, seed):
        ''' initialize surrogate with init_N randomly generated samples '''
        __counter, __max_acc_count = 0, int(1e4)
        random_state = ensure_rng(seed)
        while len(self.surrogate_buffer) < init_N:
            _state = State(if_init = True)
            for _ in range(EPISODE_LEN):
                _state = State(previous_state = _state, action = _state.generate_random_action(random_state))
            _x = _state.get_composition()
            _x_key = State.encode_key(_x)
            if _x_key not in self.surrogate_buffer:
                self.surrogate_buffer.add(_x_key)
                self.surrogate_buffer_list.append(_x)

            # loop out with error
            __counter += 1
            if __counter >= __max_acc_count:
                assert False, 'Potential permanent forloop!'

        ''' train a GPR model with the latest experimented xs, ys '''
        self.update_surrogate()

    def update_surrogate(self):
        ''' update surrogate '''
        train_x = [State.decode_key(_x_key) for _x_key in self.surrogate_buffer]
        train_y = [self.func(_x) for _x in train_x]
        train_x, train_y = np.array(train_x), np.array(train_y)

        # update self.best_score
        _best_idx = np.argmax(train_y.reshape(-1))
        self.best_score = train_y.reshape(-1)[_best_idx]
        self.best_x = train_x[_best_idx]

        ''' sklearn gpr '''
        self.surrogate = GaussianProcessRegressor(
            kernel = Matern(nu=2.5),
            alpha = 1e-6,
            normalize_y = True,
            n_restarts_optimizer = 5,
            random_state = self._random_state,
        )
        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.surrogate.fit(train_x, train_y)

        if self.enable_ei:
            self.ei_acqf = UtilityFunction(kind = "ei", xi = 0.0)

        ''' reset cached prediction '''
        self.cached_surrogate_pred_dict = dict()

    def surrogate_predict(self, x):
        _x_key = State.encode_key(x)
        if _x_key not in self.cached_surrogate_pred_dict:
            x = np.atleast_2d(x).reshape(1, -1)
            if not self.enable_ei:
                pred_val = self.surrogate.predict(x)[0]
            else:
                pred_val = _ei = self.ei_acqf.utility(x, gp = self.surrogate, y_max = self.best_score)
            self.cached_surrogate_pred_dict[_x_key] = pred_val
            return pred_val
        else:
            return self.cached_surrogate_pred_dict[_x_key]

    def update_surrogate_buffer(self, sample_xs):
        ''' update experimented xs and ys '''
        sample_xs_arr = np.array(sample_xs)
        for _x in sample_xs_arr:
            _x_key = State.encode_key(_x)
            assert not self.check_collided(_x), \
                f'Logic leak, using duplicate xs that are already in the surrogate_buffer: {_x.tolist()}\n{sample_xs_arr}\n{self.surrogate_buffer_list}'
            self.surrogate_buffer.add(_x_key)
            self.surrogate_buffer_list.append(_x)

    def check_collided(self, sample_x):
        _x_key = State.encode_key(sample_x)
        return (_x_key in self.surrogate_buffer)

    def reset(self):
        ''' 
            For RL, using a fixed initial point to start exploration is reasonable. 
            Think of it as an empty chess board.
        '''
        return State(if_init = True)
    
    def step(self, state: State, action_idx: int):
        '''
            step (步进)
            Using internally maintained GPR model to calculate reward.

            NOTE: Current implementation uses immediately calculated im.R.
            TODO: Use lazily calculated im.R if time complexity scales up. 
            TODO: 完善区分on-the-fly和surr reward的逻辑
        '''
        next_state = State(previous_state = state, action = ALL_ACTIONS[action_idx])
        ''' direct final reward '''
        if next_state.done():
            next_state_comp = next_state.get_composition()
            reward = self.func(next_state_comp)      # NOTE direct final reward
            self.surrogate_buffer.add(State.encode_key(next_state_comp))
            self.surrogate_buffer_list.append(next_state_comp)
            self.best_score = max(self.best_score, reward)
            # print(next_state_comp, round(reward, 4), round(self.best_score, 4))   # for debug
        else:
            reward = 0.
        return next_state, reward, next_state.done()
    
    def update_interaction_stat(self, state: State, state_score):
        state_comp = state.get_composition()
        self.surrogate_buffer.add(State.encode_key(state_comp))
        if sum([COMP_MIN_LIMITS[_i] <= state_comp[_i] <= COMP_MAX_LIMITS[_i] for _i in range(len(state_comp))]) == len(state_comp):
            self.best_score = max(self.best_score, state_score)

    def sample_action(self, state: State):
        return ACTIONS_TO_INDEX_DICT[state.generate_random_action()]
    
    def get_best_score(self):
        return self.best_score
    
    def get_best_x(self):
        return self.best_x
    
    def get_exp_number(self):
        assert len(self.surrogate_buffer) == len(self.surrogate_buffer_list), \
            'Something went wrong, len(surrogate_buffer) != len(surrogate_buffer_list)' + \
            f'{len(self.surrogate_buffer)}, {len(self.surrogate_buffer_list)}'
        return len(self.surrogate_buffer) - self.init_N
    
if __name__ == '__main__':
    env = Environment(
        init_N = 30,
        random_seed = 2,
    )
    print(env.get_best_score())
    print(env.get_best_x())
    # print(env.get_exp_number())
    # print(env.surrogate_buffer)