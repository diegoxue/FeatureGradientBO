import itertools
from typing import List

import numpy as np

from environment import (
    COMP_LIMITS, COMPOSITION_INTERVAL, COMPOSITION_ROUNDUP_DIGITS, TI_MIN, CompositionLimit
    
)

for elem_n in range(11, 12):
    ni_elem_idx = [2, 4, 6, 7, 8, 9, 10]
    ni_elem_n = len(list(filter(lambda x: x < elem_n, ni_elem_idx)))
    ni_comp_limits = np.array(
        (np.array(COMP_LIMITS) / COMPOSITION_INTERVAL).round(),
        dtype = int,
    )[ni_elem_idx[:ni_elem_n]]

    ti_elem_idx = [3, 5]
    ti_elem_n = len(list(filter(lambda x: x < elem_n, ti_elem_idx)))
    ti_comp_limits = np.array(
        (np.array(COMP_LIMITS) / COMPOSITION_INTERVAL).round(),
        dtype = int,
    )[ti_elem_idx[:ti_elem_n]]

    ''' 
        Combination number dictionary.
        parts_remain*elem_n_remain -> combo_num
        partx_max should in descending order
    '''
    ni_combo_num_dict = dict()
    ti_combo_num_dict = dict()

    '''  '''
    def encode(a_list: List[int]):
        return '*'.join(map(str, a_list))

    def get_combo_num(parts_remain: int, elem_n_remain: int, comp_limits: np.ndarray, combo_num_dict: dict):
        _key = encode([parts_remain, elem_n_remain])
        if _key in combo_num_dict:
            return combo_num_dict[_key]
        
        if elem_n_remain == 1:
            _sum = min(parts_remain, comp_limits[-1, 1]) + 1    # [0, ..., min(parts_remain, comp_limits[-1, 1])]
        else:
            _sum = np.float64(0)
            _min_parts = max(comp_limits[- elem_n_remain, 0], parts_remain - comp_limits[- elem_n_remain + 1:, 1].sum())
            _max_parts = min(comp_limits[- elem_n_remain, 1], parts_remain - comp_limits[- elem_n_remain + 1:, 0].sum())
            
            for _current_part in range(_min_parts, _max_parts + 1):
                _sum += get_combo_num(parts_remain - _current_part, elem_n_remain - 1, comp_limits, combo_num_dict)
        
        combo_num_dict[_key] = _sum
        return _sum

    total_remains = round(1.0 / COMPOSITION_INTERVAL)
    _ti_min, _ti_max = 200, 550,
    _ni_min, _ni_max = 200, 550

    _sum = 0
    for _ti in np.arange(total_remains - _ni_max, _ti_max + 1, 1).round().astype(int):
        _ni_parts_remain = total_remains - _ti - _ni_min
        _ti_parts_remain = _ti - _ti_min
        _ti_part = get_combo_num(_ti_parts_remain, ti_elem_n, ti_comp_limits, ti_combo_num_dict)
        _ni_part = get_combo_num(_ni_parts_remain, ni_elem_n, ni_comp_limits, ni_combo_num_dict)
        _sum += _ti_part * _ni_part

    print(elem_n, int(_sum))
