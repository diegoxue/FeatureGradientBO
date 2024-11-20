import math
from typing import List
from demo_enumerate import *
from scipy.optimize import minimize
from joblib import Parallel, delayed, load, dump

class GradientBayesianOptimization(AbstractBayesianOptimization):
    _inner_opt_budget = 80

    def suggest_next_x(self) -> List[float]:
        ''' generate initial guesses '''
        _rand_state = ensure_rng(self._init_seed)
        _guesses = _rand_state.uniform(
            low = self._bounds[:, 0],
            high = self._bounds[:, 1],
            size = (self._inner_opt_budget, self._dim)
        )
        ''' optimize using initial guesses '''
        def _to_minimize(x: np.ndarray):
            x = x.reshape(-1, self._dim)
            _mean, _std = self.predict(x)
            return - self._acqf_func(_mean, _std).item()
        
        _local_continuous_opt_x = []
        for _x in _guesses:
            _res = minimize(
                fun = _to_minimize,
                x0 = _x,
                method = 'L-BFGS-B',
                bounds = self._bounds
            )

            if not _res.success:
                continue

            _local_continuous_opt_x.append(_res.x)

        ''' round up '''
        _uniq_set = set()
        _interval = round(abs(self._x_vals[1] - self._x_vals[0]), round_digits)
        for _x in _local_continuous_opt_x:
            _supp_x_vals = []
            for _i, _x_sub in enumerate(_x):
                x_low_neighbor = math.floor(_x_sub / _interval) * _interval
                x_high_neighbor = math.ceil(_x_sub / _interval) * _interval
                _supp_x_vals.append(np.unique(np.clip([x_low_neighbor, x_high_neighbor], *(self._bounds[_i]))).round(round_digits))
            
            for _discrete_x in list(itertools.product(*_supp_x_vals)):
                _discrete_x = self.round(_discrete_x)
                _uniq_set.add(self.encode(_discrete_x))

        _candidates = np.array(list(
            map(
                lambda x: list(map(float, x.split('*'))),
                list(_uniq_set)
            )
        )).round(round_digits)

        ''' find max un-contained '''
        _mean, _std = self.predict(_candidates)
        _ei_vals = self._acqf_func(_mean, _std)

        _sorted_idx = reversed(np.argsort(_ei_vals))
        for _idx in _sorted_idx:
            _x = _candidates[_idx]
            if not self.contains(_x):
                return _x
            
        logger.error('Error. No result for inner argmax using gradient optimization')

if __name__ == '__main__':
    def _to_parallel(seed):
        ebo = GradientBayesianOptimization(dim = 2, 
                                           init_seed = seed,
                                           init_rand_num = 8,
                                           total_exp_num = 50)
        ebo.run()

        dump(
            (ebo._exp_x_history, ebo._exp_target_history), 
            f'grad_{ebo._init_rand_num}_{ebo._total_exp_num}_{seed}_{str(uuid.uuid4())[:8]}.pkl'
        )
    
    _to_parallel(0)
    Parallel(n_jobs = 1)(delayed(_to_parallel)(seed) for seed in range(1))
    