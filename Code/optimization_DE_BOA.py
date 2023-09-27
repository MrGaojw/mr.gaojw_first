from enum import Enum
from typing import List, Tuple

import numpy as np
import pandas as pd
import copy
from bayes_opt import BayesianOptimization
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.functions import single_obj as fx

from swmmtool.solver import ModifyInfo, select_best, AbstractPlan, RuleModifyInfo, SectionInfo
from swmmtool.util import mkdir, read_object, save_object, ObservedInfo, mkdir_parent
import itertools as it



# 贝叶斯
class Bayesian(Optimizer):
    def __init__(self, fun: Fun, init_points=10, n_iter=3):
        super().__init__(fun)
        self.n_iter = n_iter
        self.init_points = init_points
        ps_str = ",".join([f"x{i}" for i in range(len(fun.params))])  # 表示位置参数
        _f = eval(f"lambda {ps_str},self=self:self.fun({ps_str})")  # 接受位置参数并调用self.fun
        self.optimizer = BayesianOptimization(f=_f,
                                              pbounds={f"x{pi}": param.p_bound for pi, param in enumerate(fun.params)},
                                              random_state=1)

    def run(self):
        self.optimizer.maximize(
            init_points=self.init_points,
            n_iter=self.n_iter
        )


class GridSearchUnivariate(Optimizer):
    def __init__(self, fun: Fun, tol=0.0001):
        super().__init__(fun)
        self.tol = tol

    def run(self):
        axes_values = [[None, *np.linspace(*p.p_bound, p.p_discrete_num)] for p in self._fun.params]
        start_point = [None] * len(axes_values)
        max_objective_old = -float("inf")
        while True:
            max_objective_tem = -float("inf")
            for axis_i, values in enumerate(axes_values):
                objectives = []
                for val in values:
                    pt = start_point[:]
                    pt[axis_i] = val
                    objective = self.fun(*pt)
                    objectives.append(objective)
                # print(f"objectives={objectives}")
                val = values[np.argmax(objectives)]
                max_objective_tem = max(max(objectives), max_objective_tem)
                start_point[axis_i] = val
            if abs(max_objective_tem - max_objective_old) <= self.tol:
                break
            max_objective_old = max_objective_tem
        # select_best(self._fun.gp.output_folder_path)



class DifferentialEvolution(Optimizer):
    """
    差分进化算法
    pop_size:种群大小
    mut:变异率
    crossp:交叉概率
    maxiter:最大迭代次数
    tol:收敛容差
    """

    def __init__(self, fun: Fun, pop_size=10, mut=0.8, crossp=0.7, maxiter=50, tol=0.01):
        super().__init__(fun)
        self.pop_size = pop_size
        self.mut = mut
        self.crossp = crossp
        self.maxiter = maxiter
        self.tol = tol
        self.bounds = np.array([param.p_bound for param in fun.params])

    def run(self):
        # Initialize population 初始化
        pop = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.pop_size, len(self.bounds)))
        fitness = np.array([self.fun(*p) for p in pop])

        for i in range(self.maxiter):
            for j in range(self.pop_size):
                # Randomly select three solutions from population 从种群中随机选择三个解
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

                # Generate trial solution 生成试验解
                mutant = np.clip(a + self.mut * (b - c), self.bounds[:, 0], self.bounds[:, 1])

                # Crossover 交叉
                cross_points = np.random.rand(len(self.bounds)) < self.crossp
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, len(self.bounds))] = True
                trial = np.where(cross_points, mutant, pop[j])

                # Evaluate fitness of trial solution 评估试验解的适应度
                f = self.fun(*trial)

                # Update population if trial solution is better 如果试验解更优则更新种群
                if f > fitness[j]:
                    pop[j], fitness[j] = trial, f

            # Check convergence 检查收敛性
            if np.std(fitness) < self.tol:
                break

        # Select best solution 选择最优解
        best_idx = np.argmax(fitness)
        best_params = pop[best_idx]
        best_fitness = fitness[best_idx]
        print(f"Best DE_solution: {best_params}, DE_Fitness: {best_fitness}")
        return best_params, best_fitness


class DifferentialEvolutionWithBayesian(Optimizer):
    """
    差分进化算法+贝叶斯优化
    pop_size:种群大小
    mut:变异因子
    crossp:交叉概率
    maxiter:最大迭代次数
    tol:收敛容差
    n_iter_bayes：贝叶斯开始迭代次数，如果n_iter_bayes=10，则在每进行10次循环迭代后就会执行一次贝叶斯优化算法。
    n_iter_bayes_1:前70%的迭代次数内贝叶斯开始迭代次数，如果n_iter_bayes_1=5，则在每进行5次循环迭代后就会执行一次贝叶斯优化算法。
    n_iter_bayes_2:70%后的迭代次数内贝叶斯开始迭代次数，如果n_iter_bayes_2=2，则在每进行2次循环迭代后就会执行一次贝叶斯优化算法。
    """

    def __init__(self, fun: Fun, pop_size=20, mut=0.8, crossp=0.7, maxiter=100, tol=0.01, n_iter_bayes_1=5,
                 n_iter_bayes_2=2):
        super().__init__(fun)
        self.pop_size = pop_size
        self.mut = mut
        self.crossp = crossp
        self.maxiter = maxiter
        self.tol = tol
        self.bounds = np.array([param.p_bound for param in fun.params])
        # self.n_iter_bayes = n_iter_bayes
        self.n_iter_bayes_1 = n_iter_bayes_1
        self.n_iter_bayes_2 = n_iter_bayes_2
        self.use_bayes_opt = True

    def run(self):
        # Initialize population 初始化
        pop = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.pop_size, len(self.bounds)))
        fitness = np.array([self.fun(*p) for p in pop])

        for i in range(self.maxiter):
            for j in range(self.pop_size):
                # Randomly select three solutions from population 从种群中随机选择三个解
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

                # Generate trial solution 生成试验解
                mutant = np.clip(a + self.mut * (b - c), self.bounds[:, 0], self.bounds[:, 1])

                # Crossover 交叉
                cross_points = np.random.rand(len(self.bounds)) < self.crossp
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, len(self.bounds))] = True
                trial = np.where(cross_points, mutant, pop[j])

                # Evaluate fitness of trial solution 评估试验解的适应度
                f = self.fun(*trial)

                # Update population if trial solution is better 如果试验解更优则更新种群
                if f > fitness[j]:
                    pop[j], fitness[j] = trial, f

            # Check convergence 检查收敛性
            if np.std(fitness) < self.tol:
                break

            # Use Bayesian optimization for local optimization 使用贝叶斯优化算法进行局部优化
            if (i + 1) % self.n_iter_bayes_1 == 0 and i + 1 <= 0.7 * self.maxiter:
                # print("Use Bayesian optimization")
                bounds_dict = {f'x{i + 1}': (self.bounds[i][0], self.bounds[i][1]) for i in range(len(self.bounds))}
                optimizer = BayesianOptimization(lambda **params: -self.fun(*params.values()), bounds_dict)
                optimizer.maximize(init_points=10, n_iter=10)
                best_params_bayes = np.array(
                    [v for k, v in sorted(optimizer.max['params'].items(), key=lambda item: item[0])])
                best_fitness_bayes = -optimizer.max['target']
                best_params_bayes = np.clip(best_params_bayes, self.bounds[:, 0], self.bounds[:, 1])
                best_fitness = self.fun(*best_params_bayes)

                # Update population if Bayesian optimization finds better solution 对比贝叶斯和差分进化结果
                if best_fitness > np.max(fitness):
                    pop[np.argmax(fitness)] = best_params_bayes
                fitness[np.argmax(fitness)] = best_fitness

                # Return best solution and fitness 返回最佳解和适应度
                best_idx = np.argmax(fitness)
                best_solution = pop[best_idx]
                best_fitness = fitness[best_idx]
                return best_solution, best_fitness

            elif (i + 1) % self.n_iter_bayes_2 == 0 and i + 1 > 0.7 * self.maxiter:
                # print("Use Bayesian optimization")
                bounds_dict = {f'x{i + 1}': (self.bounds[i][0], self.bounds[i][1]) for i in range(len(self.bounds))}
                optimizer = BayesianOptimization(lambda **params: -self.fun(*params.values()), bounds_dict)
                optimizer.maximize(init_points=10, n_iter=10)
                best_params_bayes = np.array(
                    [v for k, v in sorted(optimizer.max['params'].items(), key=lambda item: item[0])])
                best_fitness_bayes = -optimizer.max['target']
                best_params_bayes = np.clip(best_params_bayes, self.bounds[:, 0], self.bounds[:, 1])
                best_fitness = self.fun(*best_params_bayes)

                # Update population if Bayesian optimization finds better solution 对比贝叶斯和差分进化结果
                if best_fitness > np.max(fitness):
                    pop[np.argmax(fitness)] = best_params_bayes
                fitness[np.argmax(fitness)] = best_fitness

                # Return best solution and fitness 返回最佳解和适应度
                best_idx = np.argmax(fitness)
                best_solution = pop[best_idx]
                best_fitness = fitness[best_idx]
                return best_solution, best_fitness
