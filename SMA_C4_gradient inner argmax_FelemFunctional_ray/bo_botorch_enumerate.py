from copy import deepcopy
from typing import Dict, List, Tuple
import uuid
import os

import joblib
import numpy as np
from loguru import logger

from environment import (COMPOSITION_ROUNDUP_DIGITS, COMPOSITION_INTERVAL,
                         TI_MIN, TI_MAX, NI_MIN, NI_MAX, CU_MIN, CU_MAX,
                         HF_MIN, HF_MAX)
from composition_bo import AbstractCompositionBayesianOptimization

class EnumerationBayesianOptimization(AbstractCompositionBayesianOptimization):
    _default_comp_block_size = 10000
    _default_all_possible_comp_path = 'data\\all_possible_comps.npy'
    
    def cal_elemental_feature(self, comp: List[float]) -> List[float]:
        comp = np.array(comp)
        return np.dot(comp, self._sel_elem_features)
    
    def suggest_next_x(self) -> List[float]:
        ''' Inner argmax of acquisition function by enumeration of all possible compositions '''
        _ei_vals = []
        for _comp_block in np.array_split(self._all_possible_comps, 
                                          np.ceil(len(self._all_possible_comps) / self._default_comp_block_size)):
            _cal_feature = self.cal_elemental_feature(_comp_block)
            _mean, _std = self.predict(_cal_feature)
            _ei_vals.extend(self.ei_acqf(_mean, _std))
        
        # find the most promising un-tested comp
        _desc_idxs = np.argsort(_ei_vals)[::-1]
        for _i in _desc_idxs:
            _comp = self._all_possible_comps[_i]
            if not self.contains(_comp):
                return _comp
        
        raise Exception('Big logic leak, should never run this line.')

    def run(self):
        ''' 0. pre-allocate all possible compositions '''
        self.preallocate_possible_comps()

        super().run()

    def preallocate_possible_comps(self):
        ''' Depth first search to pre-allocate all possible compositions '''
        if os.path.isfile(self._default_all_possible_comp_path):
            self._all_possible_comps = np.load(self._default_all_possible_comp_path)
        else:
            self._all_possible_comps = []
            _current_comp = []

            self.recursive_enumeration(_current_comp)
            self._all_possible_comps = np.array(self._all_possible_comps)

            np.save(self._default_all_possible_comp_path, self._all_possible_comps)

    def recursive_enumeration(self, current_comp: List[float],):
        '''
            Efficiently enumerate all possible compositions using DFS algorithm.

            Parameters
            ----------
            current_comp: List[float]
                current comp stack
        '''
        _idx = len(current_comp)
        if _idx == 0:
            _c_low = np.max((TI_MIN, 1.0 - NI_MAX))
            _c_high = np.min((TI_MAX, 1.0 - NI_MIN))
        else:
            _c_low = self._bounds[_idx][0]
            _c_high = self._bounds[_idx][1]
        for c in np.linspace(_c_low, _c_high, round((_c_high - _c_low) / COMPOSITION_INTERVAL + 1)).round(COMPOSITION_ROUNDUP_DIGITS):
            current_comp.append(c)
            ''' substitute '''
            if _idx == 0:
                current_comp.append(round(1.0 - current_comp[0], COMPOSITION_ROUNDUP_DIGITS))
            elif _idx == 2:
                current_comp[1] = round(current_comp[1] - c, COMPOSITION_ROUNDUP_DIGITS)
            elif _idx == 3:
                current_comp[0] = round(current_comp[0] - c, COMPOSITION_ROUNDUP_DIGITS)
            else:
                assert False
            
            if len(current_comp) < self._elem_num:
                self.recursive_enumeration(current_comp)
            else:
                self._all_possible_comps.append(deepcopy(current_comp))
            
            ''' inverse substitute '''
            current_comp.pop()
            if _idx == 0:
                current_comp.pop()
            elif _idx == 2:
                current_comp[1] = round(current_comp[1] + c, COMPOSITION_ROUNDUP_DIGITS)
            elif _idx == 3:
                current_comp[0] = round(current_comp[0] + c, COMPOSITION_ROUNDUP_DIGITS)
            else:
                assert False

if __name__ == '__main__':
    def _to_parallel(random_seed):
        enumeration_BO = EnumerationBayesianOptimization(init_seed = random_seed)
        enumeration_BO.run()
        enumeration_BO.save()

    ''' serial '''
    # _to_parallel(0), exit()

    ''' parallel '''
    _exp_num = 96
    joblib.Parallel(n_jobs = 6)(joblib.delayed(_to_parallel)(random_seed) for random_seed in range(_exp_num))
    
    
    
    
