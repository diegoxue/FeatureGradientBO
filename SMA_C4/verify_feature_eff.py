"""
Feature Effectiveness Verification Script for Shape Memory Alloy Property Prediction

This script implements an validation experiment to quantitatively verify
the effectiveness of elemental features compared to raw composition data
for predicting shape memory alloy properties.

The experimental design uses a cross-validation approach with Gaussian Process Regression
models to compare prediction accuracy between:
1. Models trained with raw elemental compositions
2. Models trained with selected weighted-average elemental features

The script implements repeated random subsampling with genetic algorithm-based
feature selection to provide statistical evidence of feature effectiveness.
"""
from copy import deepcopy
import random
from typing import Callable, Tuple
import uuid
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

from sklearn import preprocessing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import RegressorMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, RBF, WhiteKernel, Matern, ExpSineSquared, RationalQuadratic
)

from joblib import Parallel, delayed

from model_env import N_ELEM, N_ELEM_FEAT, N_ELEM_FEAT_P1, N_PROC
from model_env import CnnDnnModel
from feature_selection_ga import *

def set_seed(seed):
    """
    Set random seeds for reproducibility across multiple libraries
    
    Ensures that random operations in numpy, pytorch, and python's random
    module all produce the same results for the same seed.
    
    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set global random seed
set_seed(0) # default 0

# Generate a set of random seeds for multiple train-validation splits
seeds = np.random.randint(0, 9999, (9999, ))

def load_data():
    """
    Load and preprocess the SMA dataset for feature effectiveness verification
    
    Loads composition data, processing conditions, property measurements, and
    elemental features. Filters the data to include only samples with standard
    processing conditions (0% cold rolling, 1273K annealing, 1hr time).
    
    Returns:
        Tuple containing:
        - comp_data: Composition data for each sample (atomic fractions)
        - proc_data: Processing condition data (deformation, annealing temp/time)
        - prop_data: Property data (transformation temperatures)
        - elem_feature: Elemental features (physical/chemical properties)
    """
    # Load the default dataset
    data = pd.read_excel('data\DSC-DATA-0613.xlsx', skiprows=[1,])
    
    # Define column labels for different data types
    # Composition labels (elements in the alloy system)
    comp_labels = ['Ti', 'Ni', 'Cu', 'Hf', 'Co', 'Zr', 'Fe', 'Pd', 'Ta', 'Nb', 'V']
    
    # Processing condition labels (manufacturing parameters)
    # cold rolling, annealing temperature, annealing time
    proc_labels = ['冷轧变形量', '退火处理温度', '退火处理时间']

    # Property labels - in this case we predict, for example, Mp (martensite peak temperature)
    prop_labels = ['Mp']
    print(f'loading SMA data ...')

    # Filter out samples with missing processing conditions
    _mask = (data[proc_labels[1:]] == 0).all(axis = 1)
    data = data[~ _mask]
    print(f'deleted {sum(_mask)} items')

    # Filter to include only standard processing condition samples 
    # (0% cold rolling, 1273K annealing temperature, 1hr annealing time)
    # This isolates the effect of composition by controlling processing variables
    _mask = (data[proc_labels[0]] == 0.) & (data[proc_labels[1]] == 1273.) & (data[proc_labels[2]] == 1)
    data = data[_mask]
    print(f'deleted {sum(~ _mask)} items')
    print(f'remains {sum(_mask)} items')

    # Extract different data types into numpy arrays
    comp_data = data[comp_labels].to_numpy()
    proc_data = data[proc_labels].to_numpy()
    prop_data = data[prop_labels].to_numpy().ravel()  # Flatten to 1D array

    # Load elemental features (physical/chemical properties of elements)
    elem_feature = pd.read_excel('data\\sma_element_features.xlsx')
    elem_feature = elem_feature[comp_labels].to_numpy()  # transpose: column for each elemental feature, row for each element 

    # Return: (num_samples, num_elements), (num_samples, num_proc), (num_samples, num_prop), (num_elements, num_elem_features,)
    return comp_data, proc_data, prop_data, elem_feature

def get_gpr():
    """
    Create a Gaussian Process Regression model with appropriate kernel
    
    GPR is used because it provides uncertainty estimates and handles
    small datasets well, which is common in materials science.
    
    Returns:
        Configured GPR model instance
    """
    # Matern 5/2 kernel(differentiable functions)
    # Common choice for modeling physical processes with moderate smoothness
    kernel = 1.0 * Matern(length_scale=1.0, nu=2.5)
    
    # Return GPR model with kernel
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,  # Noise level
        normalize_y=True,  # Normalize target values
        n_restarts_optimizer=5,  # Multiple attempts to find optimal hyperparameters
        random_state=0  # For reproducibility
    )

def cv_mae(x, 
           y, 
           model: GaussianProcessRegressor, 
           split_n, 
           kf_random_state: int
           ) -> float:
    """
    Custom k-fold cross-validation with Mean Absolute Error evaluation
    
    Performs k-fold cross-validation using the provided model and data,
    and evaluates performance using Mean Absolute Error (MAE).
    
    Args:
        x: Input features (either compositions or elemental features)
        y: Target property values
        model: Gaussian Process Regression model
        split_n: Number of folds for cross-validation
        kf_random_state: Random seed for fold splitting
        
    Returns:
        Mean Absolute Error across all test folds
    """
    assert len(x) == len(y)

    # Create deep copies to avoid modifying originals
    x = deepcopy(x)
    y = deepcopy(y)
    model = clone(model)  # Create fresh copy of model

    # Scale features using robust scaling (less sensitive to outliers)
    scaler = preprocessing.RobustScaler()
    x = scaler.fit_transform(x)
    
    # Create k-fold splitter with shuffling for randomization
    kf = KFold(split_n, shuffle=True, random_state=kf_random_state)

    # Collect all test predictions and true values
    y_test_buff, y_pred_buff = [], []
    for train_index, test_index in kf.split(x):
        # Split data into train and test for this fold
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train model and make predictions
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Store results for later evaluation
        y_test_buff += y_test.tolist()
        y_pred_buff += y_pred.tolist()

    # Calculate overall MAE across all test folds
    return mean_absolute_error(y_test_buff, y_pred_buff)

def par_cv_mae(x, 
               y,  
               random_seed_list: List[int],
               n_split = 10,
               ) -> float:
    """
    Parallel execution of cross-validation with multiple random seeds
    
    Runs the cross-validation procedure multiple times with different random seeds
    to obtain a statistically robust measure of model performance.
    
    Args:
        x: Input features
        y: Target property values
        random_seed_list: List of random seeds for multiple CV runs
        n_split: Number of folds for each cross-validation
        
    Returns:
        List of MAE values from all CV runs
    """
    # Run cross-validation in parallel for each random seed
    return Parallel(n_jobs=-1)(delayed(cv_mae)(x, 
                                               y, 
                                               get_gpr(),
                                               n_split, 
                                               rs) for rs in random_seed_list)

if __name__ == '__main__':
    # Main experimental procedure to verify feature effectiveness

    # 1. Prepare data - load compositions and elemental features
    comp_data, _, prop_data, elem_feature = load_data()
    # Calculate weighted average features (dot product of compositions and elemental features)
    wavg_feature = np.dot(comp_data, elem_feature.T)

    # Storage for experimental results
    comp_res_buff, feat_res_buff = [], []
    
    # Run multiple independent experiments (64) to ensure statistical significance
    for _ in range(64):
        # 2. Randomly select 100 samples for each experiment
        selected_index = np.random.choice(len(comp_data), 100, replace=False)
        sel_comp_data = comp_data[selected_index]
        sel_wavg_feature = wavg_feature[selected_index]
        sel_prop_data = prop_data[selected_index]

        # 3. Use genetic algorithm to select the best 4 weighted average features
        # This mimics the feature selection process in the BO framework
        env = Env(sel_wavg_feature, sel_prop_data)
        ga = FeatureSelectionGaActor(env, verbose=1)
        ga.generate(n_pop=50, cxpb=0.8, mutxpb=0.1, ngen=50)
        sel_f_idx_list = ga.get_fitest_ind().f_idx_list

        # 4. Evaluate model performance using both raw compositions and selected features
        comp_to_eval = sel_comp_data  # Raw composition data
        wavg_to_eval = (sel_wavg_feature.T[sel_f_idx_list]).T  # Selected weighted average features

        # Create random seeds for cross-validation runs
        random_seeds_list = [np.random.randint(0, 999) for _ in range(64)]
        
        # Run cross-validation with both input representations and store results
        comp_res_buff.append(par_cv_mae(comp_to_eval, sel_prop_data, random_seeds_list))
        feat_res_buff.append(par_cv_mae(wavg_to_eval, sel_prop_data, random_seeds_list))

        # Save intermediate results after each experiment
        joblib.dump(
            (comp_res_buff[-1], feat_res_buff[-1]),
            f'feat_eff_res-{str(uuid.uuid4())[:8]}.pkl'
        )
    
    # Save all experimental results
    joblib.dump(
        (comp_res_buff, feat_res_buff),
        'feat_eff_res.pkl'
    )
    # Results can be analyzed to determine if selected elemental features
    # provide statistically significant improvement over raw compositions