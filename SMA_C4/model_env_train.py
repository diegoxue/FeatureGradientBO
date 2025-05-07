"""
Neural Network Model Training Module for Shape Memory Alloy Property Prediction

This module implements the training pipeline for the CNN-DNN hybrid model that predicts
properties of shape memory alloys (SMAs) based on elemental composition, elemental features, and processing conditions.

Reference: 2023_npj Computational Materials_A neural network model for high entropy alloy design

The module provides functions for:
1. Data loading and preprocessing
2. Train-validation splitting
3. Model training and evaluation
4. Model saving and loading

The implementation uses PyTorch for the neural network training and scikit-learn
for data preprocessing and evaluation metrics.
"""
import random
from typing import Callable, Tuple
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from model_env import N_ELEM, N_ELEM_FEAT, N_ELEM_FEAT_P1, N_PROC
from model_env import CnnDnnModel

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
    Load and preprocess the SMA dataset from Excel files
    
    Loads composition data, processing conditions, property measurements, and
    elemental features. Filters out samples with missing processing conditions.
    
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
    # cold rolling deformation, annealing temperature, annealing time
    proc_labels = ['冷轧变形量', '退火处理温度', '退火处理时间']

    # Property labels - in this case we're just predicting Mp (martensite peak temperature)
    # Full set: ['enthalpy（Heating）', 'Ms', 'Mp', 'Mf', 'As', 'Ap', 'Af']
    prop_labels = ['Mp']
    print(f'loading SMA data ...')

    # Filter out samples with missing processing conditions
    _mask = (data[proc_labels[1:]] == 0).all(axis = 1)
    data = data[~_mask]
    print(f'deleted {sum(_mask)} items')

    # Extract different data types into numpy arrays
    comp_data = data[comp_labels].to_numpy()
    proc_data = data[proc_labels].to_numpy()
    prop_data = data[prop_labels].to_numpy()

    # Load elemental features (physical/chemical properties of elements)
    elem_feature = pd.read_excel('data\\sma_element_features.xlsx')
    elem_feature = elem_feature[comp_labels].to_numpy()  # transpose: column for each elemental feature, row for each element 

    # Return: (num_samples, num_elements), (num_samples, num_proc), (num_samples, num_prop), (num_elements, num_elem_features,)
    return comp_data, proc_data, prop_data, elem_feature

def fit_transform(data_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,]):
    """
    Standardize all input and output data
    
    Fits StandardScalers to the data and transforms the data to have
    zero mean and unit variance, which improves neural network training.
    
    Args:
        data_tuple: Tuple containing comp_data, proc_data, prop_data, elem_feature
        
    Returns:
        Tuple containing:
        - Transformed data (same structure as input)
        - Fitted scalers for later inverse transformations
    """
    comp_data, proc_data, prop_data, elem_feature = data_tuple
    # Create a scaler for each data type
    comp_data_scaler, proc_data_scaler, prop_data_scaler, elem_feature_scaler = \
        [StandardScaler() for _ in range(len(data_tuple))]
    
    # Transform all data types
    comp_data = comp_data_scaler.fit_transform(comp_data)
    proc_data = proc_data_scaler.fit_transform(proc_data)
    prop_data = prop_data_scaler.fit_transform(prop_data)
    
    """ 
    Elemental feature standardization requires special handling:
    - Input elem_feature: (num_elem_features, num_elements), as defined in the EXCEL file
    - Output elem_feature: (num_elements, num_elem_features)
    
    Since sklearn scaler works column-wise, we transpose to calculate the mean and standard
    deviation of each element feature (e.g., VEC) across different elements.
    """
    elem_feature = elem_feature_scaler.fit_transform(elem_feature.T)

    # Return the transformed data and the scalers
    return (
        (comp_data, proc_data, prop_data, elem_feature,),
        (comp_data_scaler, proc_data_scaler, prop_data_scaler, elem_feature_scaler,),
    )

class CustomDataset(Dataset):
    """
    Custom PyTorch dataset for SMA data
    
    Stores composition, processing condition, and property data
    for batch processing during model training.
    
    Attributes:
        comp: Standardized composition data
        proc: Standardized processing condition data
        prop: Standardized property data
    """
    def __init__(self, 
                 data_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray,],
                 scaler: TransformerMixin = None): # Scaler kept for backward compatibility
        self.data_tuple = data_tuple
        self.scaler = scaler

        # Unpack data
        self.comp = self.data_tuple[0]
        self.proc = self.data_tuple[1]
        self.prop = self.data_tuple[2]
        
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.comp)
    
    def __getitem__(self, idx):
        """Return a single sample by index"""
        _comp = self.comp[idx]
        _proc = self.proc[idx]
        _prop = self.prop[idx]
        
        return _comp, _proc, _prop

def get_dataloader(data_tuple, batch_size = 16) -> DataLoader:
    """
    Create a DataLoader for batch processing during training
    
    Handles the reshaping of data into the format expected by the CNN-DNN model
    and creates batches for efficient training.
    
    Args:
        data_tuple: Tuple containing comp_data, proc_data, prop_data, elem_feature
        batch_size: Number of samples per batch
        
    Returns:
        DataLoader that yields batches of formatted data
    """
    comp_data, proc_data, prop_data, elem_feature = data_tuple
    # Create dataset (without using scaler in __getitem__)
    dataset = CustomDataset((comp_data, proc_data, prop_data,), None)

    # Prepare elemental features tensor
    # Shape: (batch_size, 1, number_of_elements, number_of_elemental_features)
    _elem_feature_tensor = torch.tensor(elem_feature, dtype=torch.float32).reshape(1, 1, *(elem_feature.shape))

    def _collate_fn(batch):
        """
        Collate function for the DataLoader
        
        Transforms a batch of samples into properly formatted tensors for the model.
        Ensures all tensors have the correct shape for the CNN architecture.
        
        Args:
            batch: List of samples from the dataset
            
        Returns:
            Tuple of tensors ready for the model
        """
        comp, proc, prop = zip(*batch)
        # Reshape compositions for CNN input - add channel and feature dimensions
        comp = torch.tensor(np.vstack(comp), dtype=torch.float32).reshape(-1, 1, comp_data.shape[-1], 1)
        # Reshape processing conditions for CNN input
        proc = torch.tensor(np.vstack(proc), dtype=torch.float32).reshape(-1, 1, proc_data.shape[-1], 1)
        # Reshape property data for loss calculation
        prop = torch.tensor(np.vstack(prop), dtype=torch.float32).reshape(-1, 1, prop_data.shape[-1], 1)

        # Clone and expand elemental features tensor to match batch size
        _elem_feature_tensor_clone = _elem_feature_tensor.expand(len(comp), 1, *(elem_feature.shape)).clone().detach()
        _elem_feature_tensor_clone.requires_grad_(False)  # No gradients needed for elemental features

        return comp, proc, prop, _elem_feature_tensor_clone

    return DataLoader(dataset, batch_size = batch_size, collate_fn = _collate_fn, shuffle = True)

def train_validate_split(data_tuple, ratio_tuple = (0.8, 0.1, 0.1)):
    """
    Split the dataset into training, validation, and test sets
    
    Implements a two-stage splitting process to create three non-overlapping sets
    with the specified proportions.
    
    Args:
        data_tuple: Tuple containing comp_data, proc_data, prop_data, elem_feature
        ratio_tuple: Tuple of (train_ratio, val_ratio, test_ratio)
        
    Returns:
        Tuple of (train_data, val_data, test_data), each containing the same structure as data_tuple
    """
    # Use a random seed from the pre-generated array
    _random_seed = next(iter(seeds))
    comp_data, proc_data, prop_data, elem_feature = data_tuple
    
    # First split: training vs. (validation + test)
    _ratio_1 = sum(ratio_tuple[1:]) / sum(ratio_tuple)
    comp_train, comp_tmp, proc_train, proc_tmp, prop_train, prop_tmp = \
        train_test_split(comp_data, proc_data, prop_data, test_size = _ratio_1, random_state = _random_seed)
    
    # Second split: validation vs. test
    _ratio_2 = ratio_tuple[2] / sum(ratio_tuple[1:])
    comp_val_1, comp_val_2, proc_val_1, proc_val_2, prop_val_1, prop_val_2 = \
        train_test_split(comp_tmp, proc_tmp, prop_tmp, test_size = _ratio_2, random_state = _random_seed)
    
    # Return three data tuples: training, validation, and test
    return (comp_train, proc_train, prop_train, elem_feature,), \
            (comp_val_1, proc_val_1, prop_val_1, elem_feature,), \
            (comp_val_2, proc_val_2, prop_val_2, elem_feature,)

def validate(model: CnnDnnModel, data_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,]) -> float:
    """
    Evaluate model performance on a validation dataset
    
    Calculates the R² score, which measures the proportion of variance in the
    target variable that is predictable from the input variables.
    
    Args:
        model: Trained CnnDnnModel
        data_tuple: Validation data tuple
        
    Returns:
        R² score (higher is better, max 1.0)
    """
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()
    
    # Create a DataLoader with batch size = entire dataset
    dl = get_dataloader(data_tuple, len(data_tuple[0]))
    
    # Get single batch containing all validation data
    comp, proc, prop, elem_t = next(iter(dl))
    
    # Make predictions (no gradient calculation needed)
    out = model(comp, elem_t, proc).detach().numpy()
    
    # Reshape ground truth to match predictions
    prop = prop.reshape(*(out.shape)).detach().numpy()
    
    # Calculate and return R² score
    return r2_score(prop, out)

def validate_a_model(num_training_epochs = 2000,
                     batch_size = 16,
                     save_path = None):
    """
    Train and validate a model with detailed performance logging
    
    Trains a model while tracking validation performance at each epoch,
    useful for analyzing model convergence and overfitting.
    
    Args:
        num_training_epochs: Number of complete passes through the training data
        batch_size: Number of samples per training batch
        save_path: Path to save validation metrics
        
    Returns:
        Tuple of (trained_model, data_tuple, scalers)
    """
    # Initialize model with default architecture
    model = CnnDnnModel()
    
    # Load and preprocess data
    d = load_data()
    d, scalers = fit_transform(d)

    # Split data into training and validation sets
    train_d, val_d_1, val_d_2 = train_validate_split(d, (0.8, 0.1, 0.1))
    
    # Define loss function and create DataLoader
    loss_fn = torch.nn.MSELoss()
    dl = get_dataloader(train_d, batch_size)
    
    # Training and validation loop
    epoch_log_buffer = []
    for epoch in range(num_training_epochs):
        # Training phase
        model.train()
        _batch_loss_buffer = []
        
        for comp, proc, prop, elem_t in dl:
            # Forward pass
            out = model(comp, elem_t, proc)
            prop = prop.reshape(*(out.shape))
            l = loss_fn(out, prop)

            # Backward pass and optimization
            model.optimizer.zero_grad()
            l.backward()
            model.optimizer.step()
            
            # Record batch loss
            _batch_loss_buffer.append(l.item())
        
        # Calculate average training loss for the epoch
        _batch_mean_loss = np.mean(_batch_loss_buffer)
        
        # Validation phase - calculate R² scores
        val_1_r2 = validate(model, val_d_1)
        val_2_r2 = validate(model, val_d_2)
        
        # Record epoch metrics
        epoch_log_buffer.append((epoch, _batch_mean_loss, val_1_r2, val_2_r2))
        
        # Print progress every 5 epochs
        if epoch % 5 == 0:
            print(epoch, _batch_mean_loss, val_1_r2, val_2_r2)
    
    # Save validation metrics if path is provided
    if save_path:
        np.savetxt(
            save_path,
            np.array(epoch_log_buffer),
            fmt = '%.6f',
            delimiter = '\t',
        )
    
    return model, d, scalers

def train_a_model(num_training_epochs = 500,
                     batch_size = 16,
                     save_log = True):
    """
    Train a model on the entire dataset for deployment
    
    Unlike validate_a_model, this function uses all available data for training
    rather than holding out validation data. Used for the final model that will
    be deployed for predictions.
    
    Args:
        num_training_epochs: Number of complete passes through the training data
        batch_size: Number of samples per training batch
        save_log: Whether to save training loss history
        
    Returns:
        Tuple of (trained_model, data_tuple, scalers)
    """
    # Initialize model with default architecture
    model = CnnDnnModel()
    
    # Load and preprocess data
    d = load_data()
    d, scalers = fit_transform(d)

    # Use all data for training (no validation split)
    train_d = d
    
    # Define loss function and create DataLoader
    loss_fn = torch.nn.MSELoss()
    dl = get_dataloader(train_d, batch_size)
    
    # Training loop
    epoch_log_buffer = []
    for epoch in range(num_training_epochs):
        # Set model to training mode
        model.train()
        _batch_loss_buffer = []
        
        # Process mini-batches
        for comp, proc, prop, elem_t in dl:
            # Forward pass
            out = model(comp, elem_t, proc)
            prop = prop.reshape(*(out.shape))
            l = loss_fn(out, prop)

            # Backward pass and optimization
            model.optimizer.zero_grad()
            l.backward()
            model.optimizer.step()
            
            # Record batch loss
            _batch_loss_buffer.append(l.item())
        
        # Calculate average training loss for the epoch
        _batch_mean_loss = np.mean(_batch_loss_buffer)
        epoch_log_buffer.append((epoch, _batch_mean_loss))
        
        # Print progress every 25 epochs
        if epoch % 25 == 0: 
            print(epoch, _batch_mean_loss)
    
    # Save training metrics if requested
    if save_log:
        np.savetxt(
            'train_err_log.txt',
            np.array(epoch_log_buffer),
            fmt = '%.6f',
            delimiter = '\t',
        )
    
    return model, d, scalers

def get_model(default_model_pth = 'model.pth',
              default_data_pth = 'data.pth',
              resume = False):
    """
    Get a trained model, either by loading from disk or training a new one
    
    This function is the main entry point for external modules that need
    to use the SMA property prediction model.
    
    Args:
        default_model_pth: Path to saved model weights
        default_data_pth: Path to saved data and scalers
        resume: Whether to load an existing model or train a new one
        
    Returns:
        Tuple of (model, data_tuple, scalers)
    """
    if resume:
        # Load existing model and data
        model = CnnDnnModel()
        model.load_state_dict(torch.load(default_model_pth))
        d, scalers = joblib.load(default_data_pth)
    else:
        # Train a new model
        model, d, scalers = train_a_model()
        # Save model weights and data for future use
        torch.save(model.state_dict(), default_model_pth)
        joblib.dump((d, scalers), default_data_pth)
    
    return model, d, scalers

if __name__ == '__main__':
    # Train and save different models for different properties
    # get_model('en_model.pth', 'en_data.pth')

    # Validate model with detailed performance tracking
    # validate_a_model()
    validate_a_model(num_training_epochs = 1000, save_path='mp_validate_log.txt')