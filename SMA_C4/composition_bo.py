''' 
Abstract Bayesian Optimization Framework for Materials Composition Design

This module implements a framework for Bayesian optimization of multicomponent alloys,
focusing on shape memory alloys (SMAs). It provides abstract classes and shared functions
for conducting materials composition design with surrogate models and efficient search strategies.

The framework integrates feature engineering approaches with Bayesian optimization techniques
to navigate the complex composition-property landscape efficiently.
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

import ray
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
    COMPOSITION_ROUNDUP_DIGITS, NI_MAX, TI_MAX, Environment, State, 
    COMP_LIMITS, ELEM_N, COMPOSITION_INTERVAL, EPISODE_LEN, COMP_MAX_LIMITS, COMP_MIN_LIMITS,
    CU_MIN, CU_MAX, HF_MIN, HF_MAX,
    get_ground_truth_func, get_mo_ground_truth_func, os_path
)
from feature_selection_ga import Env, FeatureSelectionGaActor
from feature_functionals import (weight_avg_func, delta_func, max_func, min_func,
                                range_func, maxc_func, minc_func, rangec_func)

def ignore_all_warnings(func):
    """
    Decorator to suppress all warnings in a function
    
    Useful for functions involving Gaussian Process fitting that may
    generate convergence warnings.
    
    Args:
        func: The function to wrap
        
    Returns:
        Wrapped function that suppresses warnings
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore all warnings
            return func(*args, **kwargs)
    return wrapper

def ensure_rng(random_state = None):
    """
    Creates a random number generator based on an optional seed
    
    This function ensures consistent random number generation for reproducibility.
    
    Args:
        random_state: None for a new unseeded RNG, an integer for a seeded RNG,
                     or an existing numpy.random.RandomState instance
    
    Returns:
        numpy.random.RandomState: A random number generator
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state

class AbstractCompositionBayesianOptimization(ABC):
    """
    Abstract base class for Bayesian optimization of materials compositions
    
    This class provides the core functionality for Bayesian optimization of multicomponent
    alloys using feature-based surrogate models. Concrete (inherited) implementations should define
    specific methods for suggesting new experimental points and calculating features.
    """
    def __init__(self,
                 elem_num: int = 4,
                 init_rand_num: int = 40,
                 total_exp_num: int = 100,
                 init_seed: int  = 0,
                 ):
        """
        Initialize the Bayesian optimization framework
        
        Args:
            elem_num: Number of elements in the alloy system
            init_rand_num: Number of initial random samples
            total_exp_num: Total number of experiments to run
            init_seed: Random seed for reproducibility
        """
        # Initialize ground truth function (simulates real experiments)
        self._ground_truth_func = get_mo_ground_truth_func()
        self._gpr = None  # Gaussian Process Regression model (initialized later)
        self._gpr_noise_level = 1e-6  # Noise level for GPR model
        
        # Composition space parameters
        self._elem_num = elem_num  # Number of elements in the alloy
        self._bounds = np.array(COMP_LIMITS[:elem_num])  # Compositional bounds for each element
        
        # Experiment parameters
        self._init_rand_num = init_rand_num  # Number of initial random samples
        self._total_exp_num = total_exp_num  # Total experiment budget
        
        # Element names and features
        self._all_elem_names = ['Ti', 'Ni', 'Cu', 'Hf', 'Co', 'Zr', 'Fe', 'Pd', 'Ta', 'Nb', 'V']
        
        # Load elemental features from database
        _all_elem_features = pd.read_excel(os_path('data\\sma_element_features.xlsx'))
        self._elem_features = _all_elem_features[self._all_elem_names[:elem_num]].to_numpy()
        
        # Feature selection parameters (initialized in feature_selection method)
        self._sel_elem_feat_idxs = None  # Selected elemental feature indices
        self._sel_elem_features = None  # Selected elemental features
        self._sel_elem_feat_num = 4  # Number of features to select
        
        # Acquisition function setup
        self._acqf_func = self.ei_acqf  # Default to Expected Improvement
        self._norm_dist = Normal(0, 1)  # Standard normal distribution for acquisition functions
        
        # Containers for experiments and results
        self._container = set()  # Set of evaluated compositions (for fast lookup)
        self._exp_comp_history = list()  # History of evaluated compositions
        self._exp_target_history = list()  # History of measured property values
        
        # Initialize with random compositions
        self._init_seed = init_seed
        self.init_random_compositions(init_rand_num)
        # self.init_orthogonal_compositions()  # Alternative initialization method
        
        # For storing best results
        self._best_score = float('-inf')
        self._best_x = None
        self._X = []
        self._Y = []
        self._feat_X = []
        self._max_iter = 0
        
    def init_random_compositions(self, num: int):
        """
        Initialize random compositions as initial dataset for the surrogate model
        using rejection sampling.
        
        Samples random compositions from the design space and evaluates them
        to build an initial dataset for the surrogate model.
        
        Args:
            num: Number of random compositions to generate
        """
        _rand_comps = self.rejection_sample_compositions(num=num, seed=self._init_seed)
        for _comp in _rand_comps:
            _targ = self._ground_truth_func(_comp)
            self.register(_comp, _targ)
    
    def init_orthogonal_compositions(self):
        """
        Initialize with an orthogonal design of compositions
        
        Creates a grid-based sampling of the composition space to ensure
        good coverage of the design space, focusing on key compositional ranges.
        """
        _comp_buff = []
        # Create a grid of compositions for Ti, Cu, Hf
        combos = itertools.product(
            np.linspace(0.45, 0.55, 3),  # Ti range
            np.linspace(CU_MIN, CU_MAX, 3),  # Cu range
            np.linspace(HF_MIN, HF_MAX, 3),  # Hf range
        )
        for _ti, _cu, _hf in combos:
            # Calculate Ni to maintain sum = 1.0
            _comp_buff.append([
                _ti - _hf,  # Ti adjusted for Hf
                1.0 - _ti - _cu,  # Ni as balance
                _cu,  # Cu
                _hf  # Hf
            ])
        # Evaluate all compositions in the grid
        for _comp in self.round(_comp_buff):
            _targ = self._ground_truth_func(_comp)
            self.register(_comp, _targ)

    def rejection_sample_compositions(self, 
                                      num: int, 
                                      tolerance = 1e-6, 
                                      seed = None):
        """
        Generate random compositions using rejection sampling
        
        Creates valid compositions that satisfy all constraints (element bounds and sum=1.0)
        using a rejection sampling approach.
        
        Args:
            num: Number of compositions to generate
            tolerance: Numerical tolerance for sum=1.0 constraint
            seed: Random seed for reproducibility
            
        Returns:
            List of valid composition arrays
        """
        random_state = ensure_rng(seed)

        # Pre-allocate legal values for each element
        buffered_actions = []
        for i in range(self._elem_num):
            _min, _max = COMP_MIN_LIMITS[i], COMP_MAX_LIMITS[i]
            # Create discrete steps within the range at COMPOSITION_INTERVAL precision
            _actions = np.linspace(_min, _max, round(((_max - _min) / COMPOSITION_INTERVAL)) + 1)\
                .round(COMPOSITION_ROUNDUP_DIGITS)
            buffered_actions.append(_actions)
        
        _buff_set = set()  # For detecting duplicate compositions
        _comp_buff = []  # Buffer to store valid compositions
        
        # Generate compositions until we have enough
        while len(_comp_buff) < num:
            _comp = np.zeros(self._elem_num)
            _flag = True  # Valid composition flag
            
            # Sample values for all but the last element
            for i in range(self._elem_num - 1):
                _c = random_state.choice(buffered_actions[i])
                # Check if adding this composition would exceed 1.0
                if _comp.sum() + _c > 1.0 + tolerance:
                    _flag = False  # Invalid composition
                    break
                _comp[i] = _c
                
            # Calculate the last element to make sum = 1.0
            _last_c = round(1 - _comp.sum(), COMPOSITION_ROUNDUP_DIGITS)
            # Check if the last element is within bounds
            if _last_c < COMP_MIN_LIMITS[-1] or _last_c > COMP_MAX_LIMITS[-1]:
                _flag = False  # Invalid composition
            else:
                _comp[-1] = _last_c
                
            # If composition is valid, add it to the buffer
            if _flag:
                _key = self.encode(_comp)
                # Skip if duplicate
                if _key in _buff_set:
                    continue
                _buff_set.add(_key)
                _comp_buff.append(self.round(_comp))
        
        return _comp_buff

    def feature_selection(self):
        """
        Select the most informative elemental features and transformation functions
        
        Uses a genetic algorithm to identify the optimal subset of elemental features
        and feature transformation functions that best predict the target property.
        """
        # Debug option: manually set features
        # self._sel_elem_feat_idxs = np.array([0, 1, 2, 3])
        # self._sel_feat_func_idxs = np.array([4, 5, 6, 7])
        # return

        # Get current composition and property data
        _comps = np.array(self._exp_comp_history)
        _props = np.array(self._exp_target_history)

        # GA-based feature selection
        """
        Feature functions mapping:
            1-2:    weighted_avg    delta   
            3-5:    max     min     range           
            6-8:    maxc    minc    rangec
        """
        _feature_funcs = [weight_avg_func, delta_func, max_func, min_func,
                          range_func, maxc_func, minc_func, rangec_func]
                          
        # Calculate dimensions
        _n_comps = len(_comps)  # Number of compositions
        _n_elem_feat = len(self._elem_features)  # Number of elemental features
        _n_feat_func = len(_feature_funcs)  # Number of feature functions
        
        # Initialize material feature matrix
        # Each row is a composition, each column is a (element feature, feature function) combination
        _material_feature = np.zeros(shape=(_n_comps, _n_elem_feat * _n_feat_func))
        
        # Calculate all possible features for each composition
        for i in range(_n_comps):
            for j in range(_material_feature.shape[-1]):
                """
                Matrix organization:
                - Same row: same composition
                - Same column: same (elemental feature, feature function) combination
                
                We first iterate through all elemental features for a given feature function,
                then move to the next feature function.
                """
                _elem_feat_idx = j % _n_elem_feat  # Which elemental feature
                _feat_func_idx = j // _n_elem_feat  # Which feature function
                
                # Get the specific elemental feature and function
                _tmp_feat_feat = self._elem_features[_elem_feat_idx, :]
                _tmp_feat_func = _feature_funcs[_feat_func_idx]
                
                # Calculate the material feature value
                _material_feature[i, j] = _tmp_feat_func(_comps[i], _tmp_feat_feat)

        # Create environment for genetic algorithm
        env = Env(_material_feature, _props)
        # Initialize GA actor (distributed computation)
        fs_actor = FeatureSelectionGaActor.remote(env)
        # Run genetic algorithm to select optimal features
        ray.get(fs_actor.generate.remote(
            n_pop=96,  # Population size
            cxpb=0.8,  # Crossover probability 
            mutxpb=0.1,  # Mutation probability
            ngen=200,  # Number of generations
        ))
        # Get the fittest individual (best feature combination)
        best_ind = ray.get(fs_actor.get_fitest_ind.remote())

        # Extract selected feature indices from GA results
        self._sel_elem_feat_idxs = best_ind.f_idx_list % _n_elem_feat
        self._sel_feat_func_idxs = best_ind.f_idx_list // _n_elem_feat

    def round(self, comp: List[float]) -> List[float]:
        """
        Round composition values to specified precision
        
        Ensures compositions are quantized to the defined COMPOSITION_INTERVAL
        for consistent comparison and evaluation.
        
        Args:
            comp: Raw composition array
            
        Returns:
            Rounded composition array
        """
        if not isinstance(comp, np.ndarray):
            comp = np.array(comp)
        return comp.round(COMPOSITION_ROUNDUP_DIGITS)

    def encode(self, comp: List[float]) -> str:
        """
        Encode a composition array into a string key
        
        Used for uniqueness check and storage of compositions.
        
        Args:
            comp: Composition array
            
        Returns:
            String representation of the composition
        """
        return '*'.join(map(str, self.round(comp)))

    def contains(self, comp: List[float]) -> bool:
        """
        Check if a composition has already been evaluated
        
        Args:
            comp: Composition to check
            
        Returns:
            Boolean indicating if the composition is in the history
        """
        return self.encode(comp) in self._container

    def register(self, comp: List[float], target: float):
        """
        Register a new composition-property pair
        
        Adds the pair to the history and updates internal state.
        
        Args:
            comp: Composition array
            target: Measured or predicted property value
        """
        _comp = self.round(comp)
        _key = self.encode(_comp)
        if _key not in self._container:
            self._container.add(_key)
            self._exp_comp_history.append(_comp)
            self._exp_target_history.append(target)
        else:
            logger.warning(f"Composition {_comp} already registered.")

    @staticmethod
    def init_gpr():
        """
        Initialize a Gaussian Process Regression model
        
        Returns:
            New GPR model instance
        """
        raise NotImplementedError()

    @abstractmethod
    def suggest_next_x(self) -> List[float]:
        """
        Suggest the next composition to evaluate
        
        This is the core method that implements the acquisition function optimization
        to determine the most promising candidate composition.
        
        Returns:
            Composition array to evaluate next
        """
        pass

    @abstractmethod
    def cal_elemental_feature(self, comp: List[float]) -> List[float]:
        """
        Calculate material features from elemental composition
        
        Transform raw composition into engineered features that correlate
        with the target property.
        
        Args:
            comp: Composition array
            
        Returns:
            Array of calculated features
        """
        pass

    def run(self):
        """
        Shared function for running the complete Bayesian optimization process, 
        using the following steps:
            1. identify important elemental feature subset
            2. BO outer loop
            3. BO inner loop
        """
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

    def update_elem_feat_bounds(self, elem_feat_arr: np.ndarray):
        """
        Update bounds for elemental features
        
        Calculates min/max bounds for each feature dimension to enable normalization.
        
        Args:
            elem_feat_arr: Array of elemental features
        """
        _feat_min_bounds = elem_feat_arr.min(axis = 0)
        _feat_max_bounds = elem_feat_arr.max(axis = 0)
        
        # Handle case where min == max (avoid division by zero in normalization)
        _zero_mask = (_feat_min_bounds == _feat_max_bounds)
        _feat_max_bounds[_zero_mask] = _feat_max_bounds[_zero_mask] + 1
        
        self._elem_feat_bounds = np.vstack((_feat_min_bounds, _feat_max_bounds))

    def standardize(self, elem_feat_arr: np.ndarray) -> List[float]:
        """
        Standardize features to [0,1] range
        
        Feature normalization to improve model performance.
        
        Args:
            elem_feat_arr: Raw feature array
            
        Returns:
            Normalized feature array
        """
        return (elem_feat_arr - self._elem_feat_bounds[0]) / (self._elem_feat_bounds[1] - self._elem_feat_bounds[0])

    @ignore_all_warnings
    def fit(self, 
            elem_feat: np.ndarray, 
            target: np.ndarray):
        """
        Fit a botorch GPR model.

        Parameters
        ----------
        elem_feat: numpy array
            unnormalized elemental feature 
        target: numpy array
            target property     
        """
        assert len(elem_feat) == len(target)                        

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

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with the GP model
        
        Get mean and variance predictions for new input points.
        
        Args:
            x: Feature vectors to predict

        Returns:
            Tuple of (means, variances)
        """
        x = self.standardize(x.reshape(-1, self._sel_elem_feat_num))
        x = torch.tensor(x)
        y_posterior = self._gpr.posterior(x)
        y_mean = y_posterior.mean.detach().numpy()
        y_std = y_posterior.variance.detach().sqrt().numpy()    # std
        return y_mean.flatten(), y_std.flatten()
    
    # TODO debug
    def ei_acqf(self, mean, std, xi = 0.0):
        """
        Expected Improvement acquisition function
        
        Balances exploration and exploitation by quantifying the expected amount
        by which an evaluation will improve upon the current best observation.
        
        Args:
            mean: Predicted mean values
            std: Predicted standard deviations
            xi: Exploration parameter
            
        Returns:
            EI acquisition values
        """
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
    
    def poi_acqf(self, mean, std, xi = 0.0):
        """
        Probability of Improvement acquisition function
        
        Measures the probability that a point will yield an improvement
        over the current best observation.
        
        Args:
            mean: Predicted mean values
            std: Predicted standard deviations
            xi: Exploration parameter
            
        Returns:
            POI acquisition values
        """
        if isinstance(mean, np.ndarray):
            poi_vals = np.zeros(shape = mean.shape)
            _cdf_func = lambda x: norm.cdf(x)
        elif isinstance(mean, torch.Tensor):
            poi_vals = torch.zeros_like(mean)
            _cdf_func = lambda x: self._norm_dist.cdf(x)

        mask = std > 1e-6
        mean, std = mean[mask], std[mask]
        a = (mean - self._y_max - xi)
        z = a / std
        poi_vals[mask] = _cdf_func(z)
        return poi_vals
    
    # TODO debug
    def ucb_acqf(self, mean, std, xi = 1.96):
        """
        Upper Confidence Bound acquisition function
        
        Balances exploration and exploitation via weighted sum of mean and variance.
        
        Args:
            mean: Predicted mean values
            std: Predicted standard deviations
            xi: Exploration weight (kappa)
            
        Returns:
            UCB acquisition values
        """
        return mean + xi * std

    def save(self):
        """
        Save optimization state to disk
        
        Stores the experiment history and model state for later analysis
        or resumption.
        """
        joblib.dump(
            (self._exp_comp_history, self._exp_target_history),
            f'{str(uuid.uuid4())[:8]}-{self._init_seed}-{self._init_rand_num}-{self._total_exp_num}.pkl'
        )

    @property
    def exp_round(self):
        """
        Get current experiment round number
        
        Returns:
            Current number of experiments completed
        """
        return len(self._exp_comp_history)

    @property
    def best_so_far(self):
        """
        Get the best observed target value
        
        Returns:
            Maximum target value from all experiments
        """
        return max(self._exp_target_history)