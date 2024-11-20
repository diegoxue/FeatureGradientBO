from copy import deepcopy
import itertools
import math
from typing import Dict, List
import uuid
import warnings
import joblib
import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.util import ensure_rng
from sklearn.gaussian_process.kernels import (
    ConstantKernel, RBF, WhiteKernel, Matern, ExpSineSquared, RationalQuadratic
)
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize

from environment import (
    COMPOSITION_ROUNDUP_DIGITS, Environment, State, 
    COMP_LIMITS, ELEM_N, COMPOSITION_INTERVAL, EPISODE_LEN, COMP_MAX_LIMITS, COMP_MIN_LIMITS,
    get_ground_truth_func, get_mo_ground_truth_func
)

FLOAT_ROUND_DIGIT = 4

''' Just ignore numerous sklearn warnings '''
def ignore_warnings(func):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper

class DiscreteCompositionBO(BayesianOptimization):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def register_compositional_limits(self, 
                                      elem_bounds = COMP_LIMITS, 
                                      composition_interval = COMPOSITION_INTERVAL):
        """ Register compositional limits """  
        '''
            NOTE:   This is not a duplicate variable of self._space.bounds.
                    Will be used in acq_max func with the self-defined 
                    parameter order.
        '''
        self.elem_bounds = np.array(elem_bounds)
        self.elem_n = len(elem_bounds)
        self.comp_interval = composition_interval

        assert self.elem_n == ELEM_N
    
    def _init_discrete_rand_samples_rejection(self, num_samples, tolerance = 1e-6, seed = None):
        random_state = ensure_rng(seed)

        buffered_actions = []
        for i in range(ELEM_N):
            _min, _max = COMP_MIN_LIMITS[i], COMP_MAX_LIMITS[i]
            _actions = np.linspace(_min, _max, round(((_max - _min) / COMPOSITION_INTERVAL)) + 1)\
                .round(COMPOSITION_ROUNDUP_DIGITS)
            buffered_actions.append(_actions)
        
        _buff_set = set()
        _buff = []
        while len(_buff) < num_samples:
            _comp = np.zeros(ELEM_N)
            _flag = True
            for i in range(ELEM_N - 1):
                _c = random_state.choice(buffered_actions[i])
                if _comp.sum() + _c > 1.0 + tolerance:
                    _flag = False
                    break
                _comp[i] = _c
            _last_c = round(1 - _comp.sum(), COMPOSITION_ROUNDUP_DIGITS)
            if _last_c < COMP_MIN_LIMITS[-1] or _last_c > COMP_MAX_LIMITS[-1]:
                _flag = False
            else:
                _comp[-1] = _last_c
            if _flag:
                _key = '*'.join(map(str, _comp))
                if _key in _buff_set:
                    continue
                _buff_set.add(_key)
                _buff.append(_comp)
        
        return _buff

    def init_discrete_rand_samples(self, n_init_rand: int, seed: int, x_order_buff: List, bsf_buff: List):
        """ Initialize discrete random N points for BO to train a GPR """
        assert self.elem_bounds is not None, 'Compositional bounds is not initialized.'

        _sampled_discrete_init_pts = self._init_discrete_rand_samples_rejection(n_init_rand, seed = seed)

        for _x in _sampled_discrete_init_pts:
            ''' NOTE params is stored in sorted keys order '''
            candidate_dis = self.space.array_to_params(_x)
            if self.contains(candidate_dis):
                continue
            target = self.space.target_func(**candidate_dis)
            self.register(params = candidate_dis, target = target)
        
            x_order_buff.append(_x)
            bsf_buff.append(self.max['target'])

        assert len(self.space.params) == n_init_rand
    
    def _init_continuous_rand_samples_rejection(self, num: int) -> np.ndarray:
        """ 
            Initialize continuous random N points as seeds for inner argmax of BO,
            using rejection sampling.
        """
        random_state = self._random_state
        seeds_buffer = []
        while len(seeds_buffer) < num:
            _comp_comp = np.zeros(ELEM_N)
            _found_flag = True
            for i in range(ELEM_N - 1):
                _c = random_state.random() * (COMP_MAX_LIMITS[i] - COMP_MIN_LIMITS[i]) + COMP_MIN_LIMITS[i]
                if _comp_comp.sum() + _c > 1.0:
                    _found_flag = False
                    break
                _comp_comp[i] = _c
            _last_c = 1 - _comp_comp.sum()
            if _last_c < COMP_MIN_LIMITS[-1] or _last_c > COMP_MAX_LIMITS[-1]:
                _found_flag = False
            else:
                _comp_comp[-1] = _last_c
            if _found_flag:
                seeds_buffer.append(_comp_comp)
        return np.array(seeds_buffer)

    @ignore_warnings
    def suggest_contiuous_x(self, utility_function, constraints) -> List:
        """ Most promising point to probe next """
        if len(self._space) == 0:
            raise Exception('No initial samples')
            # return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        self._gp.fit(self._space.params, self._space.target)

        # Finding argmax of the acquisition function.
        ac = utility_function.utility
        gp = self._gp
        y_max = self._space.target.max()
        bounds = self.elem_bounds
        random_state = self._random_state

        '''
            Number of times to run scipy.minimize. The default value in bayes_opt is 10.
        '''
        n_iter = 10

        ''' Main body of continuous_x inner loop (optimization) of BO '''
        continuous_x_buff = []
        
        # randomly sampled x_val seeds
        x_seeds = self._init_continuous_rand_samples_rejection(n_iter)
        
        # objective of scipy.optimize is minimization
        to_minimize = lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max)

        for x_try in x_seeds:
            # Find the minimum of minus the acquisition function
            res = minimize(lambda x: to_minimize(x),
                            x_try,
                            bounds = bounds,
                            method = 'SLSQP',
                            constraints = constraints)

            # See if success
            if not res.success:
                continue

            tmp_continuous_x = np.clip(res.x, bounds[:, 0], bounds[:, 1])
            continuous_x_buff.append(tmp_continuous_x)

        return continuous_x_buff

    def continuous_to_discrete(self, candidate_continuous: List[float]) -> np.ndarray:
        """ Convert the continuous candidate to discrete candidates """
        assert len(candidate_continuous) == ELEM_N
        _usable_dis_f = []
        for elem_idx in range(ELEM_N):
            _x = candidate_continuous[elem_idx]
            _x_low_neighbor = math.floor(_x / self.comp_interval) * self.comp_interval
            _x_high_neighbor = math.ceil(_x / self.comp_interval) * self.comp_interval
            _dis_f = np.unique(np.clip(
                [_x_low_neighbor, _x_high_neighbor], 
                self.elem_bounds[elem_idx][0], 
                self.elem_bounds[elem_idx][1]
            ))
            _usable_dis_f.append(_dis_f)

        dis_comp_buff = []
        dis_comp_cands = np.array(list(itertools.product(*_usable_dis_f))).round(FLOAT_ROUND_DIGIT)
        for row in dis_comp_cands:
            if round(row.sum(), FLOAT_ROUND_DIGIT) == 1.:   # always be careful about float comparison
                dis_comp_buff.append(row)

        return np.array(dis_comp_buff)

    def contains(self, candidates_dis: Dict[str, List[float]]):
        """ Check if candidates_dis is in the space """
        return self.space.__contains__(self.space.params_to_array(candidates_dis))
    
# NOTE keep it mind the order of bo.space.keys
def bayes_opt_serial(n_init_rand = 300,
                     n_iter = 1500,
                     seed = 42):
    '''
        n_init_rand: int, number of initial random points
        n_iter: int, number of experimental iterations (outer loop) of BO, use < 200 for laptops
    '''
    id = str(uuid.uuid4())[:8]

    '''
        NOTE params is stored in sorted keys order
        If the key strings are not sorted, you need
        to change the internal implementation of
        self.register() and self.contains()
    '''
    x_name_space = ['a-Ti', 'b-Ni', 'c-Cu', 'd-Hf', 'e-Co', 'f-Zr', 'g-Fe', 'h-Pd', 'i-Ta', 'j-Nb', 'k-V']

    ''' ground truth function '''
    # func = get_ground_truth_func('model\\uts_model.pth', 'model\\uts_data.pth',)  # single objective ground truth function
    func = get_mo_ground_truth_func()   # multi-objective ground truth function

    ''' minimize -> maximize '''
    def to_maximize(**kwargs):
        """ x: vector of input values """
        x = np.array([kwargs[xn] for xn in x_name_space])
        return func(x)

    pbounds = dict(zip(x_name_space, COMP_LIMITS))  # NOTE

    dbo = DiscreteCompositionBO(
        f = to_maximize,
        pbounds = pbounds,
        verbose = 2,            # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state = seed,
    )

    ''' explicitly register the discrete x values, the design space infos '''
    dbo.register_compositional_limits()

    # EI utility instance
    utility = UtilityFunction(kind = "ei", xi = 0.0)

    x_order_buff = []
    bsf_buff = []

    ''' random initialization n_init_rand exps '''
    dbo.init_discrete_rand_samples(n_init_rand, seed, x_order_buff, bsf_buff)

    constraints = [
        {
            'type':'eq', 
            'fun': lambda x: x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] - 1.0, 
            # 'jac': lambda x: np.ascontiguousarray(np.array([1.0, 1.0, 1.0, 1.0, 1.0]))    # NOTE no need for Jacobian vals
        },
    ]

    ''' main loop of BO, after initial random exploration '''
    for i in range(n_init_rand, n_iter + 1):
        ''' inner optimization (implicit inner loop of BO) gives several continuous x '''
        continuous_candidate_s = dbo.suggest_contiuous_x(utility, constraints)
        
        ''' apply discretization and EI calculation '''
        discrete_candidate_s, ei_s = [], []
        for candidate_cont in continuous_candidate_s:
            _all_dis_combo = dbo.continuous_to_discrete(candidate_cont)
            discrete_candidate_s += _all_dis_combo.tolist()
            ei_s += utility.utility(
                _all_dis_combo, 
                gp = dbo._gp, 
                y_max = dbo.space.target.max()
            ).flatten().tolist()
        
        sorted_idx = np.argsort(ei_s)[::-1]

        ''' enumerate all surrounding discretized xs '''
        for _i in sorted_idx:
            candidate_dis = dict(zip(x_name_space, discrete_candidate_s[_i]))
            found = not dbo.contains(candidate_dis)
            if found:
                break
        assert found, 'no new candidate found'
        
        ''' update BO dbo '''
        target = to_maximize(**candidate_dis)
        dbo.register(params = candidate_dis, target = target)
        
        bsf_buff.append(dbo.max['target'])
        
        if i % 1 == 0:  # verbose print granularity
            print(id, 'iteration:', i, 'best_func_val:', round(dbo.max['target'], FLOAT_ROUND_DIGIT))
    
    print(id, 'done')
    joblib.dump(bsf_buff, f'bayes_opt-240720-{str(uuid.uuid4())[:8]}.pkl')
    return x_order_buff, bsf_buff

# bayes_opt_serial(n_init_rand = 300, seed = np.random.randint(0, 999)), exit()
if __name__ == '__main__':
    n_init_rand = 300
    seed_list = list(np.random.randint(0, 999) for _ in range(96))
    par_res = joblib.Parallel(n_jobs = 12)(joblib.delayed(bayes_opt_serial)(n_init_rand = n_init_rand, seed = sd, n_iter = 1500) for sd in seed_list)
    joblib.dump(par_res, f'bayes_opt-sma-discrete-max_seed-{len(seed_list)}-240719-{str(uuid.uuid4())[:8]}.pkl')