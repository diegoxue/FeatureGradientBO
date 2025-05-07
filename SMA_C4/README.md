# FeatureGradientBO

Codes related with "Leveraging Feature Gradient for Efficient Acquisition Function Maximization in Material Composition Design"

## Installation

Clone the repository
```
git clone https://github.com/wsxyh107165243/FeatureGradientBO.git
```
Navigate into the project directory
```
cd FeatureGradientBO
```

## Requirements
- Python 3.10.11
- other requirements are listed in `requirements.txt`, install via pip
```
pip install -r requirements.txt
```

## Usage

```
cd SMA_C4
```

For enumeration-based BO
```
python bo_botorch_enumerate.py
```

For gradient-based BO
```
python bo_botorch_grad_opt.py
```

## Project Structure

This project implements a novel feature gradient-based approach for Bayesian optimization of material compositions, with a focus on shape memory alloys (SMAs). Below is an overview of the key components:

### Core Components

1. **Bayesian Optimization Framework**
   - `bo_botorch_grad_opt.py`: Main script for running feature gradient-based Bayesian optimization
   - `bo_botorch_enumerate.py`: Alternative approach using enumeration-based acquisition function maximization
   - `composition_bo.py`: Abstract base class for Bayesian optimization of material compositions

2. **Features and Environment**
   - `feature_functionals.py`: Implements functional representations of elemental features for compositions
   - `feature_selection_ga.py`: Genetic algorithm for selecting optimal elemental feature set
   - `environment.py`: Contains ground truth functions for testing BO and composition space definitions
   - `model_env.py`: Handles model environment and property predictions

3. **Neural Network Models**
   - `model_env_train.py`: Script for training neural network property prediction models
   - `model/`: Directory containing trained models for predicting SMA properties

4. **Analysis Tools**
   - `analyse_state_entropy.py`: Analyzes exploration behavior through k-nearest neighbor entropy
   - `verify_feature_eff.py`: Verifies the effectiveness of selected features
   - `analyse_tsne.py`: Implements t-SNE visualization for composition exploration patterns

### Key Scripts and Their Functions

| Script | Description |
|--------|-------------|
| `bo_botorch_grad_opt.py` | Main implementation of feature gradient-based Bayesian optimization |
| `bo_botorch_enumerate.py` | Implementation of enumeration-based Bayesian optimization for comparison |
| `composition_bo.py` | Abstract base class defining the Bayesian optimization framework for materials |
| `feature_functionals.py` | Implements ways to transform elemental compositions into feature representations |
| `environment.py` | Defines the ground truth functions and composition spaces |
| `feature_selection_ga.py` | Genetic algorithm implementation for feature subset selection |
| `model_env_train.py` | Training pipeline for neural network property prediction models |
| `analyse_state_entropy.py` | Tools for analyzing exploration behavior in optimization |

## How to Run Optimization

### Gradient-Based Optimization

```bash
python bo_botorch_grad_opt.py
```

This script implements the feature gradient-based approach for acquisition function maximization

### Enumeration-Based Optimization (for comparison)

```bash
python bo_botorch_enumerate.py
```

This script uses a traditional enumeration approach to maximize the acquisition function

## Extending to Other Alloy Systems

### How to use the feature gradient approach for other types of alloys?

Implementing the feature gradient approach for a different alloy system requires creating a new class that extends the existing framework. The current implementation is tailored for Shape Memory Alloys (SMAs), but the underlying principles can be applied to other materials systems.

Here's how to implement the feature gradient approach for a new alloy system:

1. **Reuse Common Components**
   - The abstract base class `AbstractCompositionBayesianOptimization` in `composition_bo.py` provides a framework that can be reused
   - The feature gradient mechanism in `feature_functionals.py` is largely material-agnostic
   - The acquisition function optimization in `bo_grad_opt.py` is independent of the specific material system

2. **Training New Property Prediction Models**:
   - Collect or generate a dataset for your alloy system
   - Follow the pattern in `model_env_train.py` to train neural networks for your specific alloy properties
   

3. **Customize These Key Components**:
   a. **Composition Space Definition**:
      - Define the valid elements and their concentration ranges
      - Implement any HEA-specific constraints (e.g., configurational entropy thresholds), modify both AbstractCompositionBayesianOptimization and environment.py

   b. **Feature Selection**:
      - Use the genetic algorithm framework in `feature_selection_ga.py` for feature subset selection

   c. **Objective Function**:
      - Define the properties of interest for your alloy system
      - Create a ground truth function (or surrogate model) in the style of `get_mo_ground_truth_func()` in `environment.py`

   d. **Rejection Sampling**:
      - Consider domain-specific constraints for your alloy system

4. **Running Optimization**:
   - Create a script similar to the main function in `bo_botorch_grad_opt.py` that uses your new class

### Benefits of This Approach

The modular design of this framework allows you to:
1. **Reuse Core Components**: The gradient-based optimization method, sampling strategies, and Bayesian optimization loop
2. **Focus on Domain-Specific Knowledge**: You only need to implement the specific rules and constraints of your alloy system

### Practical Considerations

- **Feature Selection**: Carefully select elemental features that are physically meaningful for your specific alloy system
- **Model Validation**: Ensure your property prediction models have reasonable accuracy before using them in the optimization loop
- **Constraint Handling**: Properly encoding domain-specific constraints is crucial for successful optimization

By following this approach, you can efficiently adapt the feature gradient Bayesian optimization framework to discover promising compositions for any alloy system with conituously tunable composition values, not just SMAs.