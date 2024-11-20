from copy import deepcopy
from glob import glob
from time import time
from typing import Dict, List, Tuple
import uuid
import os

import joblib
import numpy as np
from loguru import logger

from environment import (COMPOSITION_ROUNDUP_DIGITS, COMPOSITION_INTERVAL,
                         os_path)
from composition_bo import AbstractCompositionBayesianOptimization

class EnumerationBayesianOptimization(AbstractCompositionBayesianOptimization):
    _default_comp_block_size = 10000
    _default_all_possible_comp_path = 'data\\all_possible_comps.npy'
    
    def cal_elemental_feature(self, comp: List[float]) -> List[float]:
        comp = np.array(comp)
        return np.dot(comp, self._sel_elem_features)
    
    def suggest_next_x(self) -> List[float]:
        _the_comp, _the_ei = None, 0.0
        paths = glob(os_path('data\\comps\\all_possible_comps_*.npy'))
        timer_1, counter = time(), 0
        for p in paths:
            _possible_comps = np.load(p)
            counter += len(_possible_comps)
            _ei_vals = []
            for _comp_block in np.array_split(_possible_comps, 
                                            np.ceil(len(_possible_comps) / self._default_comp_block_size)):
                _cal_feature = self.cal_elemental_feature(_comp_block)
                _mean, _std = self.predict(_cal_feature)
                _ei_vals.extend(self.ei_acqf(_mean, _std))
            
            # find the most promising un-tested comp
            _desc_idxs = np.argsort(_ei_vals)[::-1]
            for _i in _desc_idxs:
                _comp = _possible_comps[_i]
                _ei = _ei_vals[_i]
                if (not self.contains(_comp)) and (_ei > _the_ei):
                    _the_comp = _comp
                    _the_ei = _ei
                if _ei <= _the_ei:
                    break
        
        # print(f'Time consumption for {counter} compositions: {time() - timer_1}'), exit
        return _the_comp

    def np_save(self):
        _all_possible_comps = np.array(self._all_possible_comps)
        np.save(os_path(f'data\\comps\\all_possible_comps_{str(uuid.uuid4())[:8]}.npy'), _all_possible_comps)
        self._all_possible_comps.clear()

    def run(self):
        ''' 0. pre-allocate all possible compositions '''
        self.preallocate_possible_comps()

        super().run()

    def preallocate_possible_comps(self):
        ''' Depth first search to pre-allocate all possible compositions '''
        if glob(os_path('data\\comps\\all_possible_comps_*.npy')):
            pass
        else:
            self._all_possible_comps = []
            _current_comp = []

            self.recursive_enumeration(_current_comp)
            
            self.np_save()

    def recursive_enumeration(self, current_comp: List[float],):
        '''
            Efficiently enumerate all possible compositions using DFS algorithm.

            Parameters
            ----------
            current_comp: List[float]
                current comp stack
        '''
        _idx = len(current_comp)
        _c_low = round(max(
            self._bounds[_idx][0],
            1.0 - sum(current_comp) - self._bounds[_idx + 1:, 1].sum()
        ), COMPOSITION_ROUNDUP_DIGITS)
        _c_high = round(min(
            self._bounds[_idx][1],
            1.0 - sum(current_comp) - self._bounds[_idx + 1:, 0].sum(),
        ), COMPOSITION_ROUNDUP_DIGITS)
        for c in np.linspace(_c_low, _c_high, round((_c_high - _c_low) / COMPOSITION_INTERVAL + 1)).round(COMPOSITION_ROUNDUP_DIGITS):
            current_comp.append(c)
            ''' substitute '''
            if len(current_comp) < self._elem_num:
                self.recursive_enumeration(current_comp)
            else:
                self._all_possible_comps.append(deepcopy(current_comp))
                if len(self._all_possible_comps) > 100 * self._default_comp_block_size:
                    self.np_save()
            ''' inverse substitute '''
            current_comp.pop()

if __name__ == '__main__':
    def _to_parallel(random_seed):
        enumeration_BO = EnumerationBayesianOptimization(init_seed = random_seed)
        enumeration_BO.run()
        enumeration_BO.save()

    ''' serial '''
    # _to_parallel(0), exit()
    EnumerationBayesianOptimization().preallocate_possible_comps()

    ''' parallel '''
    _exp_num = 96
    joblib.Parallel(n_jobs = 48)(joblib.delayed(_to_parallel)(random_seed) for random_seed in range(_exp_num))
    
    
    
    
