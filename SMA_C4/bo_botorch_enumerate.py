"""
Enumeration-based Bayesian Optimization for Shape Memory Alloy Composition Design

This module implements a Bayesian optimization approach that uses exhaustive enumeration
of the discrete composition space to find the composition that maximizes the acquisition function. 
Instead of using gradient-based methods, this approach evaluates the acquisition function at all 
possible compositions (within a discretized space) and selects the most promising candidate.

This approach is suitable for materials design problems with a relatively small number
of elements and limited compositional space, where exhaustive enumeration is computationally
feasible.
"""
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
    """
    Enumeration-based Bayesian optimization for alloy composition design
    
    This class implements a Bayesian optimization approach that explores the entire
    discrete composition space by pre-computing all possible valid compositions and
    evaluating the acquisition function for each composition at each iteration.
    
    This exhaustive search guarantees finding the global maximum of the acquisition
    function without relying on gradient-based optimization, which can be advantageous
    in highly multimodal acquisition landscapes, and relatively small number of elements.
    """
    # Block size for processing compositions in batches to manage memory usage
    _default_comp_block_size = 10000
    # Path to save/load pre-computed composition space
    _default_all_possible_comp_path = 'data\\all_possible_comps.npy'
    
    def cal_elemental_feature(self, comp: List[float]) -> List[float]:
        """
        Calculate material features from elemental composition
        
        This implementation uses a simple linear mapping approach (dot product)
        between compositions and elemental features.
        
        Args:
            comp: Composition array or list of composition arrays
            
        Returns:
            Array of calculated features
        """
        comp = np.array(comp)
        return np.dot(comp, self._sel_elem_features)
    
    def suggest_next_x(self) -> List[float]:
        """
        Suggest the next composition to evaluate by exhaustive enumeration
        
        Evaluates the acquisition function (Expected Improvement) for all possible
        compositions in the discretized space and selects the composition with
        the highest acquisition value that hasn't been evaluated yet.
        
        Returns:
            Composition array with the highest expected improvement
        """
        # Initialize list to store acquisition function values
        _ei_vals = []
        
        # Process compositions in blocks to manage memory usage
        for _comp_block in np.array_split(self._all_possible_comps, 
                                          np.ceil(len(self._all_possible_comps) / self._default_comp_block_size)):
            # Calculate features for the current block
            _cal_feature = self.cal_elemental_feature(_comp_block)
            # Make predictions with the surrogate model
            _mean, _std = self.predict(_cal_feature)
            # Calculate acquisition function values
            _ei_vals.extend(self.ei_acqf(_mean, _std))
        
        # Sort compositions by acquisition function value (highest first)
        _desc_idxs = np.argsort(_ei_vals)[::-1]
        
        # Find the highest-valued composition that hasn't been evaluated yet
        for _i in _desc_idxs:
            _comp = self._all_possible_comps[_i]
            if not self.contains(_comp):
                return _comp
        
        raise Exception('Big logic leak, should never run this line.')

    def run(self):
        """
        Run the Bayesian optimization process
        
        Overrides the parent method to add pre-allocation of all possible
        compositions before starting the optimization loop.
        """
        # Pre-allocate all possible compositions before optimization
        self.preallocate_possible_comps()

        # Run the standard BO procedure from the parent class
        super().run()

    def preallocate_possible_comps(self):
        """
        Pre-allocate all possible compositions in the discretized space
        
        Either loads pre-computed compositions from a file or generates
        them using depth-first search enumeration of the composition space.
        """
        # Check if compositions have been pre-computed
        if os.path.isfile(self._default_all_possible_comp_path):
            # Load pre-computed compositions
            self._all_possible_comps = np.load(self._default_all_possible_comp_path)
        else:
            # Initialize empty list to store compositions
            self._all_possible_comps = []
            _current_comp = []

            # Use recursive DFS to enumerate all valid compositions
            self.recursive_enumeration(_current_comp)
            self._all_possible_comps = np.array(self._all_possible_comps)

            # Save compositions for future use
            np.save(self._default_all_possible_comp_path, self._all_possible_comps)

    def recursive_enumeration(self, current_comp: List[float],):
        """
        Recursively enumerate all possible compositions using depth-first search
        
        This algorithm navigates the composition space while maintaining the
        constraint that compositions must sum to 1.0 and satisfy element-specific
        bounds. It uses domain knowledge of the quaternary Ti-Ni-Cu-Hf system to
        efficiently generate valid compositions.
        
        Args:
            current_comp: Current partial composition being built
        """
        # Current element index
        _idx = len(current_comp)
        
        # Determine bounds for the current element
        if _idx == 0:
            # Special case for first element (Ti)
            # Lower bound is max of Ti_MIN and (1-NI_MAX) to ensure enough space for Ni
            _c_low = np.max((TI_MIN, 1.0 - NI_MAX))
            # Upper bound is min of TI_MAX and (1-NI_MIN) to ensure Ni constraint satisfied
            _c_high = np.min((TI_MAX, 1.0 - NI_MIN))
        else:
            # For other elements, use predefined bounds
            _c_low = self._bounds[_idx][0]
            _c_high = self._bounds[_idx][1]
            
        # Generate all discrete values within the bounds at COMPOSITION_INTERVAL spacing
        for c in np.linspace(_c_low, _c_high, round((_c_high - _c_low) / COMPOSITION_INTERVAL + 1)).round(COMPOSITION_ROUNDUP_DIGITS):
            # Add current element composition
            current_comp.append(c)
            
            # Apply alloy system-specific substitution rules to maintain sum=1.0
            if _idx == 0:
                # After setting Ti, add Ni as balance (Ti + Ni initially = 1.0)
                current_comp.append(round(1.0 - current_comp[0], COMPOSITION_ROUNDUP_DIGITS))
            elif _idx == 2:
                # After setting Cu, subtract from Ni to maintain sum=1.0
                current_comp[1] = round(current_comp[1] - c, COMPOSITION_ROUNDUP_DIGITS)
            elif _idx == 3:
                # After setting Hf, subtract from Ti to maintain sum=1.0
                current_comp[0] = round(current_comp[0] - c, COMPOSITION_ROUNDUP_DIGITS)
            else:
                # Should never reach here with quaternary system
                assert False
            
            # Continue DFS if not all elements have been assigned
            if len(current_comp) < self._elem_num:
                self.recursive_enumeration(current_comp)
            else:
                # Add complete composition to the list
                self._all_possible_comps.append(deepcopy(current_comp))
            
            # Backtrack: undo the substitutions to explore other branches
            current_comp.pop()
            if _idx == 0:
                current_comp.pop()  # Remove Ni as well
            elif _idx == 2:
                # Restore Ni
                current_comp[1] = round(current_comp[1] + c, COMPOSITION_ROUNDUP_DIGITS)
            elif _idx == 3:
                # Restore Ti
                current_comp[0] = round(current_comp[0] + c, COMPOSITION_ROUNDUP_DIGITS)
            else:
                assert False

if __name__ == '__main__':
    def _to_parallel(random_seed):
        """
        Function to run a single BO instance with a given random seed
        
        This function can be called in parallel to run multiple independent
        optimization trials with different initializations.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        # Create and run an enumeration-based BO instance
        enumeration_BO = EnumerationBayesianOptimization(init_seed = random_seed)
        enumeration_BO.run()
        enumeration_BO.save()

    # Serial execution (uncomment to run a single instance)
    # _to_parallel(0), exit()

    # Parallel execution of multiple BO instances
    _exp_num = 96  # Number of parallel experiments
    # Run parallel jobs with joblib (6 parallel processes)
    joblib.Parallel(n_jobs = 6)(joblib.delayed(_to_parallel)(random_seed) for random_seed in range(_exp_num))
    
    
    
    
