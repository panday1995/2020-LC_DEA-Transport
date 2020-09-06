#!/usr/bin/env python
# coding: utf-8
from collections.abc import Iterable
from itertools import product
from multiprocessing import Pool
import brightway2 as bw
import numpy as np
import os
import pandas as pd
import pickle


def value(data, year, transport_mode):
    data_series = data[data.year == year].loc[:, transport_mode]
    return data_series

def activities(data, act_type):
    # choose activities
    act_list = data[data.index == act_type]
    act_eidb_dict = {}
    act_eidb_list = []
    for i in range(len(act_list)):
        act = [
            act
            for act in eidb
            if act_list.iloc[i, 0] in act["name"]
            and act_list.iloc[i, 1] in act["location"]
        ]
        if len(act) == 0:
            continue
        else:
            act_eidb_list.append(act[0])
    act_eidb_dict[act_type] = act_eidb_list
    return act_eidb_dict

def actvity_weight_generator(data, year):
    # 构建funtional units的权重
    act_list_year = data[data[year] != 0]  # 挑选出当年的activities
    fu_dict_total = {}
    for i in range(len(act_type_list)):
        find_activity = activities(
            act_list_year, act_type_list[i]
        )  # 找到lca数据库中的activities
        if isinstance(act_list_year.loc[act_type_list[i], year], Iterable):
            activity_weight_pair = dict(
                zip(
                    find_activity[act_type_list[i]],
                    act_list_year.loc[act_type_list[i], year].values,
                )
            )
        else:
            activity_weight_pair = dict(
                zip(
                    find_activity[act_type_list[i]],
                    [act_list_year.loc[act_type_list[i], year]],
                )
            )
        fu_dict_total[act_type_list[i]] = activity_weight_pair
    return fu_dict_total


def weight_year_dict(year):
    weight_year[year] = actvity_weight_generator(activity_eidb_df, year)

def create_fu_list(weight_data, year, DMU_name):
    data_act_pair = list(zip(activity_data.columns[1:], act_type_list))
    fu_list_total = []
    for (k, v) in data_act_pair:
        act = weight_data[year][v]
        DMU_data = activity_data[activity_data["year"] == year].loc[DMU_name, k]
        if v == "road_passenger":
            for (key, value) in act.items():
                fu_list_total.append({key: DMU_data * value / 10})
        else:
            for (key, value) in act.items():
                fu_list_total.append({key: DMU_data * value})
    return fu_list_total


def multiImpactMonteCarloLCA(functional_unit, list_methods, iterations):
    # Step 1
    MC_lca = bw.MonteCarloLCA(functional_unit)
    MC_lca.lci()
    # Step 2
    C_matrices = {}
    # Step 3
    for method in list_methods:
        MC_lca.switch_method(method)
        C_matrices[method] = MC_lca.characterization_matrix
    # Step 4
    results = np.empty((len(list_methods), iterations))
    # Step 5
    for iteration in range(iterations):
        next(MC_lca)
        for method_index, method in enumerate(list_methods):
            results[method_index, iteration] = (
                C_matrices[method] * MC_lca.inventory
            ).sum()
    return results

# 导入不同运输模式的周转量数据
activity_data = pd.read_excel(
    r"E:\tencent files\chrome Download\Research\DEA\DEA_transport\Data_input\data.xlsx",
    sheet_name="activity_data",
    index_col=0,
)

# 不同运输模型的LCA名称
activity_eidb_df = pd.read_excel(
    r"E:\tencent files\chrome Download\Research\DEA\DEA_transport\Data_input\data.xlsx",
    sheet_name="lca_EF",
    index_col=0,
)

if "ecoinvent 3.6" not in bw.databases:
    link = r"E:\ecoinvent3.6cutoffecoSpold02\datasets"
    ei36 = bw.SingleOutputEcospold2Importer(link, "ecoinvent 3.6", use_mp=False)
    ei36.apply_strategies()
    ei36.statistics()
    ei36.write_database()

eidb = bw.Database("ecoinvent 3.6")
structured_array = np.load(eidb.filepath_processed())
transport_modes = list(activity_data.columns)[1:]
years = list(range(2013, 2018))

fu_value_dict = {}
for year in years:
    fu_subdict = {}
    for transport_mode in transport_modes:
        fu_subdict[transport_mode] = value(activity_data, year, transport_mode)
    fu_value_dict[year] = fu_subdict

act_type_list = list(dict.fromkeys([i for i in activity_eidb_df.index]))

weight_year={}
for year in years:
    weight_year_dict(year) 

def lcia_mc_cal(year, DMU_name, list_methods, iterations,weight_data=weight_year, ):
    fu_list = create_fu_list(weight_data, year, DMU_name)
    city_lcia_mc = []
    for i in fu_list:
        mc_results = pd.DataFrame(
            multiImpactMonteCarloLCA(i, indicators, iterations),
            index=["GWP", "FDP", "PMFP"],
        )
        city_lcia_mc.append(mc_results)
    city_mc_df = pd.concat(city_lcia_mc)
    with open(DMU_name + str(year) + ".pickle", "wb",) as file:
        pickle.dump(city_mc_df, file)
    print("data for DMU {} in year {} have finished".format(DMU_name,year))

def lcia_mc_multiprocess(year, DMU):
    lcia_mc_cal( year, DMU, indicators, 1000)


ReCiPe = [
    method
    for method in bw.methods
    if "ReCiPe Midpoint (H) V1.13" in str(method)
    and "w/o LT" not in str(method)
    and "no LT" not in str(method)
    and "obsolete" not in str(method)
]

lcia = ["GWP100", "PMFP", "FDP"]

indicators = [i for i in ReCiPe if i[2] in lcia]
os.chdir(r"E:\tencent files\chrome Download\Research\DEA\DEA_transport\Data_input")

if not os.path.exists("city_mc"):
    os.mkdir("city_mc")
os.chdir("city_mc")

DMU_list = [DMU for DMU in activity_data[activity_data["year"] == 2013].index[18:30]]


if __name__ ==  '__main__': 
    with Pool() as p:
        p.starmap(lcia_mc_multiprocess,product(years,DMU_list))
