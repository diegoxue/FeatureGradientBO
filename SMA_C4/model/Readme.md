# Neural Network Models for Shape Memory Alloy Property Prediction

This directory contains trained neural network models for predicting various properties of shape memory alloys (SMAs) based on their composition and processing conditions.

## Overview

The models in this directory are CNN-DNN hybrid networks trained to predict thermomechanical properties of Ti-Ni-Cu-Hf-based shape memory alloys. These models serve as "ground truth" in the Bayesian optimization framework for materials discovery and design.

## Contents

### Model Files
- `en_model.pth`: Model weights for enthalpy prediction
- `mp_model.pth`: Model weights for martensite peak temperature (Mp) prediction
- `ap_model.pth`: Model weights for austenite peak temperature (Ap) prediction

### Data Files
- `en_data.pth`: Data and scalers specific to enthalpy prediction
- `mp_data.pth`: Data and scalers specific to martensite transformation temperature
- `ap_data.pth`: Data and scalers specific to austenite transformation temperature

## Model Architecture

The models implement a hybrid CNN-DNN architecture that processes:
1. **Composition data**: Atomic fractions of elements in the alloy
2. **Elemental features**: Physical and chemical properties of constituent elements
3. **Processing conditions**: Manufacturing parameters (deformation, annealing temperature, annealing time)

The architecture combines convolutional layers to extract features from composition and elemental properties with fully connected layers for final property prediction.

## Usage

To use these models for property prediction:

```python
from model_env_train import get_model

# Load pre-trained model and associated data
model, data, scalers = get_model(
    default_model_pth='model/mp_model.pth',
    default_data_pth='model/mp_data.pth',
    resume=True  # Set to True to load existing model
)

# Prepare input data
comp_data = [0.5, 0.4, 0.1, 0.0]  # Example: Ti-Ni-Cu-Hf composition
proc_data = [0.0, 1273.0, 1.0]    # Example: 0% cold work, 1273K annealing, 1h time

# Create appropriate data format and tensor transformations
# (see environment.py for implementation details)

# Use model for prediction
prediction = model(comp_tensor, elem_feature_tensor, proc_tensor)

# Convert prediction back to original scale
actual_property = scalers[2].inverse_transform(prediction.detach().numpy())
```

For automated property prediction through the Bayesian optimization framework, refer to `environment.py` which provides functions `get_ground_truth_func()` and `get_mo_ground_truth_func()` that encapsulate these models.

## Model Performance

The models were trained and validated on experimental data from DSC measurements of various SMA compositions. Cross-validation performance metrics:
- **Enthalpy**: R² ~= 0.74
- **Transformation Temperatures**: R² ~= 0.91

## Training Process

These models were trained using the pipeline defined in `model_env_train.py`. The training process includes:
- Data standardization
- Train-validation splitting
- Hyperparameter optimization
- Early stopping to prevent overfitting

To retrain models with new data, refer to the training functions in `model_env_train.py`.

## References

These models follow the architecture described in:
- "A neural network model for high entropy alloy design" in NPJ Computational Materials (2023)