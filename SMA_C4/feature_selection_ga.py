"""
Genetic Algorithm for Feature Selection in Materials Informatics

This module implements a parallel genetic algorithm (GA) framework for selecting optimal 
elemental feature set for property prediction of alloys.

The GA optimizes feature selection by evolving a population of feature subsets, where:
- Each individual represents a subset of features
- Fitness is evaluated through cross-validated model performance
- Evolution occurs through selection, crossover, and mutation operations

The implementation uses Ray for distributed computation to accelerate the 
feature selection process.
"""
from __future__ import annotations
import random
import copy
import time
from typing import List, Tuple

import ray
import joblib
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.gaussian_process.kernels import (RBF, ConstantKernel, Matern,
                                              RationalQuadratic, WhiteKernel)

# Global parameters for the genetic algorithm
# Number of features to select
F_N = 4
# Number of parallel threads for computation
THREAD_N = 12
# Number of splits for cross-validation
SPLITS_N = 10
# Number of repetitions for cross-validation to ensure statistical robustness
CV_N_REPEAT = 4

def get_random_forest() -> RandomForestRegressor:
    """
    Create a RandomForestRegressor for property prediction
    
    Random Forest is chosen for its robustness, ability to handle non-linear relationships,
    and relative insensitivity to feature scaling.
    
    Returns:
        Configured RandomForestRegressor instance
    """
    return RandomForestRegressor(n_estimators=100,    # Number of trees in the forest
                                 random_state=43,)    # Fixed seed for reproducibility

def get_gpr() -> GaussianProcessRegressor:
    """
    Create a GaussianProcessRegressor for property prediction
    
    GPR is used as an alternative to RandomForest, providing uncertainty estimates
    and potentially better performance for smaller datasets.
    
    Returns:
        Configured GaussianProcessRegressor instance
    """
    # Set initial value and bounds for kernel hyperparameters
    fit_init, fit_bounds = 10., (1e-5, 1e8)
    
    # Composite kernel: combination of Matérn kernel with white noise
    # Matérn kernel is well-suited for modeling physical processes
    gpr_kernel = ConstantKernel(fit_init, fit_bounds) * \
                    Matern(fit_init, fit_bounds) + \
                    WhiteKernel(fit_init, fit_bounds)
    
    # Note: RBF kernel was tested but performed worse in practice
    # gpr_kernel = RBF(fit_init, fit_bounds) + \
    #                 WhiteKernel(fit_init, fit_bounds)
    
    # Initialize model with multiple optimizer restarts to avoid local minima
    model = GaussianProcessRegressor(kernel=gpr_kernel, 
                                     n_restarts_optimizer=100, 
                                     random_state=42)
    return model

def custom_cross_validation_ray(model, x, y) -> float:
    """
    Perform repeated k-fold cross-validation using mean absolute error
    
    This function is designed to be called in a Ray distributed computing context
    to evaluate fitness of feature subsets in parallel.
    
    Args:
        model: Machine learning model (clone will be used for each fold)
        x: Input features
        y: Target property values
        
    Returns:
        Negative mean MAE across all repetitions (negative for maximization in GA)
    """
    assert len(x) == len(y)

    # Clone model to avoid modifying the original
    model = clone(model)
    
    # Scale features using robust scaling (less sensitive to outliers)
    scaler = preprocessing.RobustScaler()
    x = scaler.fit_transform(x)
    
    # Store MAE values across multiple CV repetitions
    mae_buff = []
    
    # Repeat cross-validation multiple times with different random splits
    for _ in range(CV_N_REPEAT):
        # Create k-fold splitter with shuffling
        kf = KFold(SPLITS_N, shuffle=True)

        # Collect predictions and true values across all folds
        y_test_buff, y_pred_buff = [], []
        for train_index, test_index in kf.split(x):
            # Split data for this fold
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train model and make predictions
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Store results
            y_test_buff += y_test.tolist()
            y_pred_buff += y_pred.tolist()

        # Calculate MAE for this repetition
        mae_buff.append(mean_absolute_error(y_test_buff, y_pred_buff))
    
    # Return negative mean MAE (for maximization in fitness function)
    return -np.mean(mae_buff)

@ray.remote
def calculate_fitness_ray(model, feature_val, prop_val, feature_num, scaler, ind):
    """
    Ray remote function to calculate fitness of an individual in parallel
    
    This function extracts the selected features for an individual and evaluates
    their predictive performance using cross-validation.
    
    Args:
        model: Machine learning model to evaluate feature subset
        feature_val: Complete feature matrix
        prop_val: Target property values
        feature_num: Total number of available features
        scaler: Feature scaler
        ind: Individual (feature subset) to evaluate
        
    Returns:
        Fitness value (negative MAE)
    """
    # Get indices of selected features
    feature_idx_list = np.asarray(ind.f_idx_list, dtype=np.int32)
    
    # Validate feature indices
    assert len(feature_idx_list) == F_N
    assert len(set(feature_idx_list)) == F_N
    assert feature_idx_list.min() >= 0 and feature_idx_list.max() < feature_num
    
    # Extract selected features
    x = feature_val[:, feature_idx_list]
    # Scale features
    x = scaler.fit_transform(x)
    
    # Calculate fitness using cross-validation
    fitness = custom_cross_validation_ray(model, x, prop_val)
    
    return fitness

class Env:
    """
    Environment for feature selection
    
    This class encapsulates the data and evaluation methods needed for
    feature selection, providing methods to calculate fitness of feature subsets.
    
    Attributes:
        feature_val: Matrix of all feature values
        prop_val: Target property values
        feature_num: Total number of available features
        model: Machine learning model for evaluation
        scaler: Feature scaler
    """
    def __init__(self, wavg_feature, props) -> None:
        """
        Initialize environment with data
        
        Args:
            wavg_feature: Feature matrix (samples × features)
            props: Target property values
        """
        self.feature_val = np.array(wavg_feature)
        self.prop_val = np.array(props)
        self.feature_num = self.feature_val.shape[1]
        # Default to RandomForest model, for efficiency considerations
        self.model = get_random_forest()
        self.scaler = preprocessing.RobustScaler()

    def __custom_cross_validation(self, x, y) -> float:
        """
        Internal method for cross-validation
        
        Similar to custom_cross_validation_ray but for sequential evaluation.
        
        Args:
            x: Input features
            y: Target property values
            
        Returns:
            Negative mean MAE (for maximization in fitness function)
        """
        assert len(x) == len(y)

        model = clone(self.model)
        scaler = preprocessing.RobustScaler()
        x = scaler.fit_transform(x)
        
        mae_buff = []
        
        for _ in range(CV_N_REPEAT):
            kf = KFold(SPLITS_N, shuffle=True)

            y_test_buff, y_pred_buff = [], []
            for train_index, test_index in kf.split(x):
                X_train, X_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                y_test_buff += y_test.tolist()
                y_pred_buff += y_pred.tolist()

            mae_buff.append(mean_absolute_error(y_test_buff, y_pred_buff))
        
        return -np.mean(mae_buff)

    def calculate_fitness(self, ind: Individual):
        """
        Calculate fitness of an individual
        
        Evaluates how well the selected features predict the target property.
        
        Args:
            ind: Individual (feature subset) to evaluate
            
        Returns:
            Fitness value (negative MAE)
        """
        # For debugging: return random.random()

        # Get indices of selected features
        feature_idx_list = np.asarray(ind.f_idx_list, dtype=np.int32)

        # Validate feature indices
        assert len(feature_idx_list) == F_N
        assert len(set(feature_idx_list)) == F_N
        assert feature_idx_list.min() >= 0 and feature_idx_list.max() < self.feature_num
        
        # Extract and scale selected features
        x = self.feature_val[:, feature_idx_list]
        x = self.scaler.fit_transform(x)
        
        # Calculate fitness using cross-validation
        fitness = self.__custom_cross_validation(x, self.prop_val)
        
        return fitness
    
    @property
    def total_f_N(self) -> int:
        """
        Get total number of available features
        
        Returns:
            Number of features
        """
        return self.feature_num

class Individual:
    """
    Individual class for genetic algorithm
    
    Represents a potential solution (feature subset) in the genetic algorithm.
    Each individual encodes a specific combination of features.
    
    Attributes:
        f_idx_list: List of feature indices
        fitness: Fitness value (initialized as None)
    """
    def __init__(self, f_idx_list: List[int]):
        """
        Initialize individual with feature indices
        
        Args:
            f_idx_list: List of selected feature indices
        """
        self.f_idx_list = np.array(f_idx_list)
        self.fitness = None

def init_individual(total_f_num: int, f_num: int) -> Individual:
    """
    Generate a random individual
    
    Creates an individual by randomly selecting a subset of features.
    
    Args:
        total_f_num: Total number of available features
        f_num: Number of features to select
        
    Returns:
        New random individual
    """
    # Randomly sample f_num features from the total feature pool
    f_idx_list = random.sample(range(total_f_num), f_num)
    return Individual(f_idx_list)

def softmax(x: List[float]) -> np.ndarray:
    """
    Compute softmax values for scores
    
    Transforms raw scores into probabilities for selection.
    Subtracts max(x) for numerical stability.
    
    Args:
        x: List of scores
        
    Returns:
        Normalized probabilities
    """
    e_x = np.exp(x - np.max(x))  # subtract max(x) for numerical stability
    return e_x / e_x.sum()

def select(population: List[Individual], k: int) -> List[Individual]:
    """
    Select individuals from population for breeding
    
    Implements a roulette wheel selection (fitness proportionate selection)
    where probability of selection is proportional to fitness.
    
    Args:
        population: List of individuals
        k: Number of individuals to select
        
    Returns:
        Selected individuals
    """
    # Calculate sum of fitness values
    fitness_sum = sum(individual.fitness for individual in population)
    
    # Calculate selection probabilities proportional to fitness
    probabilities = [individual.fitness / fitness_sum for individual in population]
    
    # Normalize probabilities using softmax (handles negative fitness values)
    probabilities = softmax(probabilities)
    
    # Select k individuals without replacement based on probabilities
    sel_idx = np.random.choice(len(population), size=k, replace=False, p=probabilities)
    return [population[i] for i in sel_idx]

def mutate(parent: Individual, mutation_prob: float, total_f_num: int) -> Individual:
    """
    Mutate an individual by replacing features
    
    Creates a new individual by potentially replacing some features
    with new ones, based on mutation probability.
    
    Args:
        parent: Individual to mutate
        mutation_prob: Probability of mutating each feature
        total_f_num: Total number of available features
        
    Returns:
        Mutated individual
    """
    # Deep copy of feature indices
    f_idx_list = copy.deepcopy(parent.f_idx_list)
    
    # Determine positions to mutate based on mutation probability
    mutate_pos = [i for i in range(len(f_idx_list)) if random.random() < mutation_prob]
    
    # Select new feature indices from those not already in the individual
    mutate_val = random.sample(
        list(set(range(total_f_num)) - set(f_idx_list)), 
        len(mutate_pos)
    )
    
    # Replace selected positions with new values
    f_idx_list[mutate_pos] = mutate_val
    return Individual(f_idx_list)

def cross_over(parent_1: Individual, parent_2: Individual, crx_pb: float = 0.25) -> Tuple[Individual, Individual]:
    """
    Perform crossover between two parents
    
    Creates two new individuals by exchanging features between parents.
    The crossover operation maintains unique features in each individual.
    
    Args:
        parent_1: First parent
        parent_2: Second parent
        crx_pb: Crossover probability for each feature
        
    Returns:
        Two offspring individuals
    """
    # Deep copy parent feature lists
    f_idx_list_1 = copy.deepcopy(parent_1.f_idx_list)
    f_idx_list_2 = copy.deepcopy(parent_2.f_idx_list)
    
    # Find features unique to each parent
    only_in_1 = list(set(f_idx_list_1) - set(f_idx_list_2))
    only_in_2 = list(set(f_idx_list_2) - set(f_idx_list_1))
    
    # Shuffle unique features to introduce randomness
    random.shuffle(only_in_1)
    random.shuffle(only_in_2)
    
    # Create new feature lists by combining common features with unique ones
    new_f_idx_list_1 = only_in_1 + list(set(f_idx_list_1) & set(f_idx_list_2))
    new_f_idx_list_2 = only_in_2 + list(set(f_idx_list_1) & set(f_idx_list_2))
    
    # Exchange features between offspring with probability crx_pb
    for i in range(len(only_in_1)):
        if random.random() < crx_pb:
            new_f_idx_list_1[i], new_f_idx_list_2[i] = new_f_idx_list_2[i], new_f_idx_list_1[i]
    
    return Individual(new_f_idx_list_1), Individual(new_f_idx_list_2)

def elitism_replacement(population: List[Individual], offspring: List[Individual]):
    """
    Perform elitism replacement
    
    Combines current population and offspring, then selects the top individuals
    based on fitness to form the new population.
    
    Args:
        population: Current population
        offspring: Newly created offspring
        
    Returns:
        New population
    """
    # Combine current population and offspring
    combined = population + offspring
    
    # Sort by fitness in descending order (higher fitness is better)
    combined.sort(key=lambda ind: ind.fitness, reverse=True)
    
    # Keep only the top individuals equal to the original population size
    return combined[:len(population)]

@ray.remote
class FeatureSelectionGaActor:
    """
    Ray actor for parallelized genetic algorithm feature selection
    
    This class implements a genetic algorithm to find optimal feature subsets
    using distributed computation with Ray.
    
    The genetic algorithm includes these phases:
    1. Initialization: Create random initial population
    2. Evaluation: Assess fitness of each individual
    3. Selection: Choose parents based on fitness
    4. Crossover: Create offspring from parents
    5. Mutation: Introduce random variations
    6. Replacement: Form new population
    7. Termination: End after specified generations
    """

    def __init__(self, ff_obj: Env, verbose: int = 0):
        """
        Initialize GA actor
        
        Args:
            ff_obj: Environment object with data and evaluation methods
            verbose: Verbosity level (0 or 1)
        """
        self.verbose = verbose
        self.dominants_buffer = {}  # Store best individual from each generation
        self.best_ind = None
        
        # Validate environment
        if ff_obj is None:
            raise ValueError("Please provide a valid environment.")
        else:
            self.env = ff_obj

        # Print information if verbose
        if self.verbose == 1:
            print(
                "Will select best features among {} features.".format(self.env.feature_val.shape[1])
            )
            print("Shape of train_x: {} and target: {}".format(self.env.feature_val.shape, self.env.prop_val.shape))

    def par_eval(self, pop: List[Individual]) -> List[float]:
        """
        Evaluate population fitness in parallel
        
        Uses Ray to distribute fitness calculations across workers.
        
        Args:
            pop: Population to evaluate
            
        Returns:
            List of fitness values
        """
        # Create Ray tasks for parallel evaluation
        futures = [
            calculate_fitness_ray.remote(
                self.env.model,
                self.env.feature_val,
                self.env.prop_val,
                self.env.feature_num,
                self.env.scaler,
                ind
            ) for ind in pop
        ]
        # Get results from all tasks
        return ray.get(futures)

    def generate(self, n_pop, cxpb=0.5, mutxpb=0.2, ngen=5):
        """
        Run genetic algorithm to evolve population
        
        Main method that executes the genetic algorithm loop.
        
        Args:
            n_pop: Population size
            cxpb: Crossover probability
            mutxpb: Mutation probability
            ngen: Number of generations
            
        Returns:
            Final evolved population
        """
        # Print parameters if verbose
        if self.verbose == 1:
            print(
                "Population: {}, crossover_probablity: {}, mutation_probablity: {}, total generations: {}".format(
                    n_pop, cxpb, mutxpb, ngen
                )
            )

        # Initialize population with random individuals
        pop = [init_individual(self.env.total_f_N, F_N) for _ in range(n_pop)]

        # Evaluate initial population fitness
        print("EVOLVING.......")
        fitnesses = self.par_eval(pop)

        # Assign fitness to individuals
        for ind, fit in zip(pop, fitnesses):
            ind.fitness = fit

        # Main evolution loop
        for g in range(ngen):
            # Store best individual from current generation
            self.dominants_buffer[g] = max(pop, key=lambda ind: ind.fitness)

            # Print progress
            print(" GENERATION {} ".format(g + 1).center(25, '-'))
            print("Best fitness: {}".format(self.dominants_buffer[g].fitness))
            
            # Select individuals for breeding
            selected_pop = select(pop, len(pop) // 2)
            new_individuals = []
            
            # Apply crossover to pairs of selected individuals
            for ind_1, ind_2 in zip(selected_pop[::2], selected_pop[1::2]):
                new_individuals += list(cross_over(ind_1, ind_2, cxpb))

            # Apply mutation to selected individuals
            for ind in selected_pop:
                new_individuals.append(mutate(ind, mutxpb, self.env.total_f_N))

            # Evaluate fitness of new individuals
            fitnesses = self.par_eval(new_individuals)
            for ind, fit in zip(new_individuals, fitnesses):
                ind.fitness = fit
            print("Evaluated %i new individuals" % len(new_individuals))

            # Create new population with elitism
            pop = elitism_replacement(pop, new_individuals)                         

        print("-- Only the fittest survives --")

        # Find best individual in final population
        self.best_ind = max(pop, key=lambda ind: ind.fitness)
        print(
            "Best individual is %s, %s" % (self.best_ind, self.best_ind.fitness)
        )

        return pop

    def save_dominants_buffer(self, file_name: str):
        """
        Save best individuals from each generation
        
        Args:
            file_name: File path to save data
        """
        joblib.dump(self.dominants_buffer, file_name)

    def review_pop(self, pop: List[Individual]):
        """
        Print population statistics
        
        Calculates and prints statistics about population fitness.
        
        Args:
            pop: Population to analyze
        """
        fits = [ind.fitness for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        
        if self.verbose == 1:
            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)

    def get_fitest_ind(self) -> Individual:
        """
        Get the fittest individual from the latest generation
        
        Returns:
            Best individual
        """
        _key = max(self.dominants_buffer.keys())
        return self.dominants_buffer[_key]

if __name__ == "__main__":
    # Example usage of the genetic algorithm
    env = Env()
    ga = FeatureSelectionGaActor(env, verbose=1)
    ga.generate(n_pop=200, cxpb=0.8, mutxpb=0.1, ngen=50)
    ga.get_fitest_ind()
    ga.save_dominants_buffer('dominants_buffer.pkl')