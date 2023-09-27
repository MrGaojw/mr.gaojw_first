import matplotlib.pyplot as plt
import networkx as nx
import swmmio
from bayes_opt import BayesianOptimization
# from pyswmm.reader import TimeSeries
from swmmio import dataframe_from_inp

import sys

import logging

import os

# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swmmtool.graph import SwmmGraph
from swmmtool.optimization import Param, GridSearch, Fun, GridSearchUnivariate, Bayesian, DifferentialEvolution, \
    DifferentialEvolutionWithBayesian
from swmmtool.plot import show_param
from swmmtool.solver import GeneralPlan, ObjectiveOption, ModifyInfo, select_best, RuleModifyInfo
from swmmtool.structs import Section, Conduits, Subcatchments, Timeseries, Subareas, Infiltration
from swmmtool.util import ObservedNodeInfo, mkdir

# ==========================================================================
# ===========================手动修改=========================================
# ==========================================================================
# 观察点为id为 Y1101000009, 观察的属性为 水头 ObservedNodeInfo.VN_Head
observed_info = ObservedNodeInfo("060209-28300209-000022", ObservedNodeInfo.VN_Total_Inflow)

# 两个峰, 峰值最大误差不超过 0.25 m, 峰现时间不超过 45 分钟
option = ObjectiveOption(peak_num=2, peak_diff_tolerance=0.25, time_diff_tolerance=45,
                         penalty=0)
gp = GeneralPlan(inp_path="input1/JSZ-PSFQ.inp", output_folder_path="result",
                 observed_path="input1/rain.xlsx",
                 observed_info=observed_info, objective_option=option
                 )
qs = gp.quick_select

params = [
    # Param(ModifyInfo(Section.SUBAREAS, Subareas.N_Imperv, objs=qs.upstream_subcatchments), p_bound=(0.01, 0.04),
    #       p_discrete_num=10),
    # Param(ModifyInfo(Section.SUBAREAS, Subareas.N_Perv, objs=qs.upstream_subcatchments), p_bound=(0.1, 0.3),
    #       p_discrete_num=10),
    # Param(ModifyInfo(Section.SUBAREAS, Subareas.S_Imperv, objs=qs.upstream_subcatchments), p_bound=(0.1, 2.5),
    #       p_discrete_num=10),
    # Param(ModifyInfo(Section.SUBAREAS, Subareas.S_Perv, objs=qs.upstream_subcatchments), p_bound=(2, 10),
    #       p_discrete_num=10),
    # Param(ModifyInfo(Section.INFILTRATION, Infiltration.Param1, objs=qs.upstream_subcatchments), p_bound=(3.3, 50.8),
    #       p_discrete_num=10),
    # Param(ModifyInfo(Section.INFILTRATION, Infiltration.Param3, objs=qs.upstream_subcatchments), p_bound=(2, 7),
    #       p_discrete_num=10),
    # Param(ModifyInfo(Section.CONDUITS, Conduits.Roughness, objs=qs.upstream_edges), p_bound=(0.011, 0.03),
    #       p_discrete_num=10),
    # Param(ModifyInfo(Section.SUBCATCHMENTS, Subcatchments.Imperv, objs=qs.upstream_subcatchments), p_bound=(0, 100),
    #       p_discrete_num=10),
    # Param(ModifyInfo(Section.SUBCATCHMENTS, Subcatchments.Slope, objs=qs.upstream_subcatchments), p_bound=(0.6, 1.8),
    #       p_discrete_num=10),
    # Param(ModifyInfo(Section.INFILTRATION, Infiltration.Param2, objs=qs.upstream_subcatchments), p_bound=(0.5, 5),
    #       p_discrete_num=10),
]


def best():
    """
    选取最佳结果
    """
    select_best(gp.output_folder_path)


def grid_search():
    """
    格点遍历
    """
    opt = GridSearch(fun=Fun(gp, params))
    opt.run()


def grid_search_univariate():
    """
    坐标轮换法
    """
    # show_param(inp_path=gp.inp_path, modify_infos=[param.info for param in params],
    #            output_folder=f"{gp.output_folder_path}/param")
    opt = GridSearchUnivariate(fun=Fun(gp, params))
    opt.run()
    select_best(gp.output_folder_path)


def bayesian_optimization():
    """
    贝叶斯优化
    使用该方法时，参数设置中的 p_discrete_num 不起作用
    """
    # show_param(inp_path=gp.inp_path, modify_infos=[param.info for param in params],
    #            output_folder=f"{gp.output_folder_path}/param")
    # init_points 表示随机计算10个点
    # n_iter 表示优化的点的数量
    opt = Bayesian(fun=Fun(gp, params), init_points=10, n_iter=20)
    opt.run()
    select_best(gp.output_folder_path)


def differentialevolution():
    """
    差分进化算法
    使用该方法时，参数设置中的 p_discrete_num 不起作用
    """
    opt = DifferentialEvolution(fun=Fun(gp, params), pop_size=20, mut=0.8, crossp=0.7, maxiter=100, tol=0.01)
    opt.run()
    select_best(gp.output_folder_path)


def debo():
    """
    差分进化算法+贝叶斯优化
    使用该方法时，参数设置中的 p_discrete_num 不起作用
    """
    opt = DifferentialEvolutionWithBayesian(fun=Fun(gp, params), pop_size=20, mut=0.8, crossp=0.7, maxiter=100,
                                            tol=0.01,
                                            n_iter_bayes_1=10, n_iter_bayes_2=10)
    opt.run()
    select_best(gp.output_folder_path)


if __name__ == '__main__':
    # grid_search()
    # grid_search_univariate()
    # bayesian_optimization()
    # show_param(inp_path="input/swmm.inp", modify_infos=[param.info for param in params], output_folder="result/param")
    # differentialevolution()
    debo()
