from __future__ import annotations
import random
import copy
from typing import List, Tuple

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

''' golbal variables '''
# feature number
F_N = 4
# thread number
THREAD_N = 12
# number of splits for cross validation
SPLITS_N = 10
# repeat times 
CV_N_REPEAT = 4

def get_random_forest() -> RandomForestRegressor:
    '''
        获取RandomForestRegressor对象, 用于回归

        :return: RandomForestRegressor对象
    '''
    return RandomForestRegressor(n_estimators = 100,    # NOTE: 20240817_val = 200
                                 random_state = 43,)

def get_gpr() -> GaussianProcessRegressor:
    '''
        获取GaussianProcessRegressor对象, 用于回归

        :return: GaussianProcessRegressor对象
    '''
    fit_init, fit_bounds = 10., (1e-5, 1e8)
    gpr_kernel = ConstantKernel(fit_init, fit_bounds) * \
                    Matern(fit_init, fit_bounds) + \
                    WhiteKernel(fit_init, fit_bounds)
    ''' RBF核实际测试效果较差 '''
    # gpr_kernel = RBF(fit_init, fit_bounds) + \
    #                 WhiteKernel(fit_init, fit_bounds)    
    model = GaussianProcessRegressor(kernel = gpr_kernel, 
                                     n_restarts_optimizer = 100, 
                                     random_state = 42)
    return model

class Env:
    '''
        Environment for feature selection of strain data.
    '''
    def __init__(self, 
                 wavg_feature, 
                 proc_data,
                 props) -> None:
        self.feature_val = np.array(wavg_feature)
        self.proc_data = np.array(proc_data)
        self.prop_val = np.array(props)
        self.feature_num = self.feature_val.shape[1]
        self.model = get_random_forest()
        self.scaler = preprocessing.RobustScaler()

    def __custom_cross_validation(self, x, y) -> float:
        ''' 
            Custom 10-fold cross validation.

            :param x: feature values
            :param y: property values
            :return: r2 score
        '''
        assert len(x) == len(y)

        model = clone(self.model)
        scaler = preprocessing.RobustScaler()
        x = scaler.fit_transform(x)
        
        mae_buff = []
        
        for _ in range(CV_N_REPEAT):
            kf = KFold(SPLITS_N, shuffle = True)

            y_test_buff, y_pred_buff = [], []
            for train_index, test_index in kf.split(x):
                X_train, X_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                y_test_buff += y_test.tolist()
                y_pred_buff += y_pred.tolist()

            mae_buff.append(mean_absolute_error(y_test_buff, y_pred_buff))
        
        return - np.mean(mae_buff)

    def calculate_fitness(self, ind: Individual):
        ''''
            calculate fitness of an individual

            :param ind: individual
            :return: fitness
        '''
        # return random.random()  # for debug

        feature_idx_list = np.asarray(ind.f_idx_list, dtype = np.int32)

        assert len(feature_idx_list) == F_N
        assert len(set(feature_idx_list)) == F_N
        assert feature_idx_list.min() >= 0 and feature_idx_list.max() < self.feature_num
        
        x = self.feature_val[:, feature_idx_list]
        x = self.scaler.fit_transform(x)
        x = np.concatenate((x, self.proc_data), axis = -1)  # NOTE
        fitness = self.__custom_cross_validation(x, self.prop_val)
        
        return fitness
    
    @property
    def total_f_N(self) -> int:
        return self.feature_num

class Individual:
    """
        Individual class
        An individual represents a possible solution for the problem.
    """
    def __init__(self, f_idx_list: List[int]):
        ''' feature index list, type: List[int] '''
        self.f_idx_list = np.array(f_idx_list)
        self.fitness = None

def init_individual(total_f_num: int, f_num: int) -> Individual:
    """
        Generate an individual.

        :param total_f_num: The total number of features.
        :param f_num: The number of features to select.
        :return: An individual.
    """
    f_idx_list = random.sample(range(total_f_num), f_num)
    return Individual(f_idx_list)

def softmax(x: List[float]) -> np.ndarray:
    """
        Compute softmax values for each sets of scores in x

        :param x: List[float]
        :return: np.ndarray
    """
    e_x = np.exp(x - np.max(x))  # subtract max(x) for numerical stability
    return e_x / e_x.sum()

def select(population: List[Individual], k: int) -> List[Individual]:
    """
        Select the individuals from a population.

        轮盘赌选择(Roulette Wheel Selection)
        在这种方法中，每个个体被选中的概率与其适应度函数值的大小成正比。
        具体来说，每个个体的选择概率等于其适应度函数值除以所有个体适应度函数值的总和。
        这种方法更倾向于选择适应度高的个体，但也给适应度低的个体留下了一定的选择机会。

        :param population: The population.
        :param k: The number of individuals to select.
        :return: The selected individuals.
    """
    fitness_sum = sum(individual.fitness for individual in population)
    probabilities = [individual.fitness / fitness_sum for individual in population]
    ''' fitness defined as r2 can be negative, need to normalize it to [0, 1] '''
    probabilities = softmax(probabilities)
    sel_idx = np.random.choice(len(population), size = k, replace = False, p = probabilities)
    return [population[i] for i in sel_idx]

def mutate(parent: Individual, mutation_prob: float, total_f_num: int) -> Individual:
    """
        Mutate an individual by replacing one of its attribute with a random integer value.

        :param individual: The individual to be mutated.
        :return: A new mutated individual.
    """
    f_idx_list = copy.deepcopy(parent.f_idx_list)
    mutate_pos = [i for i in range(len(f_idx_list)) if random.random() < mutation_prob]
    mutate_val = random.sample(
        list(set(range(total_f_num)) - set(f_idx_list)), 
        len(mutate_pos)
    )
    f_idx_list[mutate_pos] = mutate_val
    return Individual(f_idx_list)

def cross_over(parent_1: Individual, parent_2: Individual, crx_pb: float = 0.25) -> Tuple[Individual, Individual]:
    """
        Cross over two individuals.

        :param individual1: The first individual.
        :param individual2: The second individual.
        :return: Two new crossed individuals.
    """
    f_idx_list_1 = copy.deepcopy(parent_1.f_idx_list)
    f_idx_list_2 = copy.deepcopy(parent_2.f_idx_list)
    only_in_1 = list(set(f_idx_list_1) - set(f_idx_list_2))
    only_in_2 = list(set(f_idx_list_2) - set(f_idx_list_1))
    ''' shuffle '''
    random.shuffle(only_in_1)
    random.shuffle(only_in_2)
    ''' cross over '''
    new_f_idx_list_1 = list(set(f_idx_list_1) & set(f_idx_list_2)) + only_in_1
    new_f_idx_list_2 = list(set(f_idx_list_1) & set(f_idx_list_2)) + only_in_2
    for i in range(len(only_in_1)):
        if random.random() < crx_pb:
            new_f_idx_list_1[i], new_f_idx_list_2[i] = new_f_idx_list_2[i], new_f_idx_list_1[i]
    return Individual(new_f_idx_list_1), Individual(new_f_idx_list_2)

def elitism_replacement(population: List[Individual], offspring: List[Individual]):
    """ 
        Perform Elitism Replacement
    
        :param population: The population.
        :param offspring: The offspring.
        :return: The new population.
    """
    combined = population + offspring
    combined.sort(key = lambda ind: ind.fitness, reverse = True)
    return combined[:len(population)]

class FeatureSelectionGA:
    """
        FeaturesSelectionGA
        This class uses Genetic Algorithm to find out the best features for the given data.

        遗传算法通常包含以下几个阶段:
        初始化(Initialization):创建一个初始种群。这个种群通常是随机生成的。
        评估(Evaluation):评估种群中每个个体的适应度。
        选择(Selection):<根据每个个体的适应度来选择用于交叉的个体。适应度高的个体有更高的机会被选中。>
        交叉(Crossover):从已经选择的个体中创建新的个体。这个过程模拟了生物的交配过程。
        突变(Mutation):对新生成的个体进行随机的小修改，这个过程模拟了生物的突变过程。
        替换(Replacement):用新生成的个体替换掉种群中的一部分或全部个体。
        终止(Termination).
    """

    def __init__(self, ff_obj: Env, verbose: int = 0):
        """
        Parameters
        -----------
        ff_obj: {object}, environment for feature selection
        verbose: 0 or 1
        """
        self.verbose = verbose
        # self.final_fitness = []
        self.dominants_buffer = {}
        self.best_ind = None
        if ff_obj == None:
            raise ValueError("Please provide a valid environment.")
        else:
            self.env = ff_obj

        if self.verbose == 1:
            print(
                "Will select best features among {} features.".format(self.env.feature_val.shape[1])
            )
            print("Shape of train_x: {} and target: {}".format(self.env.feature_val.shape, self.env.prop_val.shape))

    def par_eval(self, pop: List[Individual]) -> List[float]:
        ''' 
            parallel evaluation of fitness

            :param pop: population
            :return: fitnesses
        '''
        fitnesses = joblib.Parallel(n_jobs=THREAD_N)(joblib.delayed(self.env.calculate_fitness)(x) for x in pop)
        return fitnesses

    def generate(self, n_pop, cxpb=0.5, mutxpb=0.2, ngen=5):

        """
        Generate evolved population
        Parameters
        -----------
            n_pop : {int}
                    population size
            cxpb  : {float}
                    crossover probablity
            mutxpb: {float}
                    mutation probablity
            n_gen : {int}
                    number of generations
            set_toolbox : {boolean}
                          If True then you have to create custom toolbox before calling
                          method. If False use default toolbox.
        Returns
        --------
            Fittest population
        """

        if self.verbose == 1:
            print(
                "Population: {}, crossover_probablity: {}, mutation_probablity: {}, total generations: {}".format(
                    n_pop, cxpb, mutxpb, ngen
                )
            )

        pop = [init_individual(self.env.total_f_N, F_N) for _ in range(n_pop)]

        # Evaluate the entire population
        print("EVOLVING.......")
        fitnesses = self.par_eval(pop)

        for ind, fit in zip(pop, fitnesses):
            ind.fitness = fit

        for g in range(ngen):
            self.dominants_buffer[g] = max(pop, key = lambda ind: ind.fitness)

            print(" GENERATION {} ".format(g + 1).center(25, '-'))
            print("Best fitness: {}".format(self.dominants_buffer[g].fitness))
            # self.review_pop(pop)

            selected_pop = select(pop, len(pop) // 2)
            new_individuals = []
            # Apply crossover and mutation on the offspring
            for ind_1, ind_2 in zip(selected_pop[::2], selected_pop[1::2]):     # TODO: check cross_over order
                new_individuals += list(cross_over(ind_1, ind_2, cxpb))

            for ind in selected_pop:
                new_individuals.append(mutate(ind, mutxpb, self.env.total_f_N))

            # Evaluate the new individuals
            fitnesses = self.par_eval(new_individuals)
            for ind, fit in zip(new_individuals, fitnesses):
                ind.fitness = fit
            print("Evaluated %i new individuals" % len(new_individuals))

            # replacement
            pop = elitism_replacement(pop, new_individuals)                         

        print("-- Only the fittest survives --")

        self.best_ind = max(pop, key = lambda ind: ind.fitness)
        print(
            "Best individual is %s, %s" % (self.best_ind, self.best_ind.fitness)
        )
        # self.get_final_scores(pop, fits)

        return pop

    def save_dominants_buffer(self, file_name: str):
        """
            Save fitness in each generation in a file

            :param file_name: file name
        """
        joblib.dump(self.dominants_buffer, file_name)

    def review_pop(self, pop: List[Individual]):
        """
            Review population by statistics

            :param pop: population
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
        _key = max(self.dominants_buffer.keys())
        return self.dominants_buffer[_key]

if __name__ == "__main__":
    env = Env()
    ga = FeatureSelectionGA(env, verbose = 1)
    ga.generate(n_pop = 200, cxpb = 0.8, mutxpb = 0.1, ngen = 50)
    ga.get_fitest_ind()
    ga.save_dominants_buffer('dominants_buffer.pkl')