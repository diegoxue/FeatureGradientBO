from abc import ABC, abstractmethod
import itertools
from typing import List, Tuple
import uuid
from joblib import Parallel, delayed, dump
from loguru import logger
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from utils import *
import matplotlib.pyplot as plt
from composition_bo import ensure_rng, ignore_all_warnings
from scipy.stats import norm

round_digits = 4
func_name = 'rastrigin'
x_min, x_max = func_params[func_name]['x_min'], func_params[func_name]['x_max']
x_vals = np.linspace(x_min, x_max, func_params[func_name]['act_dim']).round(round_digits)

''' test code '''
# y_vals = [rastrigin([_x]) for _x in x_vals]
# plt.plot(x_vals, y_vals)
# plt.show()

class AbstractBayesianOptimization(ABC):
    def __init__(self,
                 dim: int = 1,
                 init_rand_num: int = 10,   # TODO
                 total_exp_num: int = 100,
                 init_seed: int  = 0,
                 ):
        _func = func_params[func_name]['func']
        self._ground_truth_func = lambda x: - _func(x)

        self._gp = GaussianProcessRegressor(
            kernel = Matern(nu=2.5),
            alpha = 1e-6,
            normalize_y = True,         # for argmax\-(x)<f_acq>, no difference
            n_restarts_optimizer = 10,
            random_state = ensure_rng(init_seed),
        )
        # self._gpr_noise_level = 1e-6    # no effect currently

        self._dim = dim
        self._bounds = np.array([[x_min, x_max] for _ in range(dim)])
        self._x_vals = x_vals
        
        self._init_rand_num = init_rand_num
        self._total_exp_num = total_exp_num

        self._acqf_func = self.ei_acqf

        self._container = set()
        self._exp_x_history = list()
        self._exp_target_history = list()

        self._init_seed = init_seed
        self.init_random_xs(init_rand_num)

    def init_random_xs(self, num: int):
        ''' Initialize `num` random x '''
        _rand_state = ensure_rng(self._init_seed)
        while len(self._exp_x_history) < self._init_rand_num:
            _x = [_rand_state.choice(self._x_vals) for _ in range(self._dim)]
            if self.contains(_x):
                continue
            _targ = self._ground_truth_func(_x)
            self.register(_x, _targ)

    def round(self, x: List[float]) -> List[float]:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return x.round(round_digits)

    def encode(self, x: List[float]) -> str:
        return '*'.join(map(str, self.round(x)))

    def contains(self, x: List[float]) -> bool:
        ''' If a x is already experimented. '''
        key = self.encode(x)
        return key in self._container

    def register(self, x: List[float], target: float):
        ''' Register the experimented point into exp history buffer '''
        x = self.round(x)
        assert not self.contains(x), self.encode(x)
        key = self.encode(x)
        self._container.add(key)
        self._exp_x_history.append(x)
        self._exp_target_history.append(target)
        self._y_max = max(self._exp_target_history)

    @abstractmethod
    def suggest_next_x(self) -> List[float]:
        pass

    @ignore_all_warnings
    def fit(self, _x, _target) -> None:
        self._gp.fit(_x, _target)

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = x.reshape(-1, self._dim)
        return self._gp.predict(x, return_std = True)

    def ei_acqf(self, mean: np.ndarray, std: np.ndarray, xi = 0.0):
        ''' Calculate expected improvement. '''      
        ei_vals = np.zeros(shape = mean.shape)
        _cdf_func = lambda x: norm.cdf(x)
        _pdf_func = lambda x: norm.pdf(x)

        mask = std > 1e-10   # EI(x\-(std == 0)) <- 0
        mean, std = mean[mask], std[mask]
        a = (mean - self._y_max - xi)
        z = a / std
        ei_vals[mask] = a * _cdf_func(z) + std * _pdf_func(z)
        return ei_vals
    
    def run(self):
        _local_id = str(uuid.uuid4())[:8]

        ''' 1. BO outer loop '''
        for exp_round in range(self.exp_round + 1, self._total_exp_num + 1):
            _x = np.array(self._exp_x_history)
            _target = np.array(self._exp_target_history).reshape(-1, 1)
            self.fit(_x, _target)
            
            ''' 3. BO inner loop '''
            _next_comp = self.suggest_next_x()
            _next_target = self._ground_truth_func(_next_comp)

            self.register(_next_comp, _next_target)

            ''' 4. log '''
            logger.info(f'{_local_id} - round {exp_round}, best-so-far {self.best_so_far}')

    @property
    def exp_round(self):
        return len(self._exp_x_history)

    @property
    def best_so_far(self):
        return self._y_max
    
class EnumerationBayesianOptimization(AbstractBayesianOptimization):
    _ei_eval_block_size = 1000

    def __init__(self, dim: int = 1, 
                 init_rand_num: int = 10, 
                 total_exp_num: int = 100, 
                 init_seed: int = 0):
        super().__init__(dim, init_rand_num, total_exp_num, init_seed)

        if dim > 2:
            logger.warning('This script will be extremely slow for "dim > 2"')

        self._all_possible_x = np.array(
            list(itertools.product(self._x_vals, repeat = self._dim))
        ).round(round_digits)

    def suggest_next_x(self):
        _ei_vals = []
        _num_blocks = np.ceil(len(self._all_possible_x) / self._ei_eval_block_size)
        for _x_block in np.array_split(self._all_possible_x, _num_blocks):
            _mean, _std = self.predict(_x_block)
            _ei_vals.extend(
                self._acqf_func(_mean, _std).flatten()
            )

        _sorted_idx = reversed(np.argsort(_ei_vals))
        for _idx in _sorted_idx:
            _x = self._all_possible_x[_idx]
            if not self.contains(_x):
                return _x
            
if __name__ == '__main__':
    def _to_parallel(seed):
        ebo = EnumerationBayesianOptimization(dim = 1, 
                                           init_seed = seed,
                                           init_rand_num = 8,
                                           total_exp_num = 50)
        ebo.run()

        dump(
            (ebo._exp_x_history, ebo._exp_target_history), 
            f'enum_{ebo._init_rand_num}_{ebo._total_exp_num}_{seed}_{str(uuid.uuid4())[:8]}.pkl'
        )

    Parallel(n_jobs = -1)(delayed(_to_parallel)(seed) for seed in range(96))