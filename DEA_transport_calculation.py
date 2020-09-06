#!/usr/bin/env python
# coding: utf-8


from multiprocessing import Pool
from itertools import product
import numpy as np
import os
import pandas as pd
import pickle
import pulp
import psutil
import time

def set_lowpriority():
    parent = psutil.Process()
    parent.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

def read_excel(iteration):
    dea_inout = pd.read_excel(
        os.path.join(dir_data, xls), sheet_name=str(iteration), header=0
    )
    return dea_inout

class DEAProblem:
    def __init__(self, inputs, outputs, bad_outs, weight_vector,  disp='weak disposability', directional_factor=None, returns='CRS',
                 in_weights=[0, None], out_weights=[0, None],badout_weights=[0, None]):
        self.inputs = inputs
        self.outputs = outputs
        self.bad_outs = bad_outs
        self.returns = returns
        self.disp = disp
        self.weight_vector = weight_vector # weight vector in directional distance function      
        
        self.J, self.I = self.inputs.shape  # no of DMUs, inputs
        _, self.R = self.outputs.shape  # no of outputs
        _, self.S = self.bad_outs.shape # no of bad outputs
        self._i = range(self.I)  # inputs
        self._r = range(self.R)  # outputs
        self._s = range(self.S)  # bad_output
        self._j = range(self.J)  # DMUs
        if directional_factor == None:
            self.gx = self.inputs
            self.gy = self.outputs
            self.gb = self.bad_outs
        else:
            self.gx = directional_factor[:self.I]
            self.gy = directional_factor[self.I:(self.I+self.J)]
            self.gy = directional_factor[(self.I+self.J):]

        
        self._in_weights = in_weights  # input weight restrictions
        self._out_weights = out_weights  # output weight restrictions
        self._badout_weights = badout_weights # bad output weight restrictions
        
        # creates dictionary of pulp.LpProblem objects for the DMUs
        self.dmus = self._create_problems()
    
    def _create_problems(self):
        """
        Iterate over the DMU and create a dictionary of LP problems, one
        for each DMU.
        """

        dmu_dict = {}
        for j0 in self._j:
            dmu_dict[j0] = self._make_problem(j0)
        return dmu_dict
    
    def _make_problem(self, j0):
        """
        Create a pulp.LpProblem for a DMU.
        """

        # Set up pulp
        prob = pulp.LpProblem("".join(["DMU_", str(j0)]), pulp.LpMaximize)
        self.weights = pulp.LpVariable.dicts("Weight", (self._j),
                                                  lowBound=self._in_weights[0])
        self.betax = pulp.LpVariable.dicts("scalingFactor_x", (self._i),
                                                  lowBound=0,upBound=1)

        self.betay = pulp.LpVariable.dicts("scalingFacotr_y", (self._r),
                                                  lowBound=0)
   
        self.betab = pulp.LpVariable.dicts("scalingFacotr_b", (self._s),
                                                  lowBound=0, upBound=1)
        
        # Set returns to scale
        if self.returns == "VRS":
            prob += pulp.lpSum([weight for weight in self.weights]) == 1

        # Set up objective function      
        prob += pulp.lpSum([(self.weight_vector[i]*self.betax[i]) for i in self._i]
                           +[(self.weight_vector[self.I+r]*self.betay[r]) for r in self._r]
                          +[(self.weight_vector[self.I+self.R+s]*self.betab[s]) for s in self._s])

        # Set up constraints
        for i in self._i:
            prob += pulp.lpSum([(self.weights[j0]*
                                              self.inputs.values[j0][i]) for j0 in self._j]) <= self.inputs.values[j0][i]-self.betax[i]*self.gx.values[j0][i]
        for r in self._r:
            prob += pulp.lpSum([(self.weights[j0]*
                                              self.outputs.values[j0][r]) for j0 in self._j]) >= self.outputs.values[j0][r]+self.betay[r]*self.gy.values[j0][r]
        
        if  self.disp == "weak disposability":  
            for s in self._s:   # weak disposability
                prob += pulp.lpSum([(self.weights[j0]*
                                                self.bad_outs.values[j0][s]) for j0 in self._j]) == self.bad_outs.values[j0][s]-self.betab[s]*self.gb.values[j0][s]
        
        elif self.disp =="strong disposability":
            for s in self._s:
                prob += pulp.lpSum([(self.weights[j0]*
                                                self.bad_outs.values[j0][s]) for j0 in self._j]) >= self.bad_outs.values[j0][s]-self.betab[s]*self.gb.values[j0][s] 
        return prob
    
    def solve(self):
        """
        Iterate over the dictionary of DMUs' problems, solve them, and collate
        the results into a pandas dataframe.
        """

        sol_status = {}
        sol_weights = {}
        sol_efficiency = {}

        for ind, problem in list(self.dmus.items()):
            problem.solve()
            sol_status[ind] = pulp.LpStatus[problem.status]
            sol_weights[ind] = {}
            for v in problem.variables():
                sol_weights[ind][v.name] = v.varValue
            sol_efficiency[ind] = pulp.value(problem.objective)
        return sol_status, sol_efficiency, sol_weights

def inout_data(iteration, data_columnslist):
    dea_data = read_excel(iteration)
    data_df = dea_data.loc[:, data_columnslist]
    return data_df

def results_df(inputs, outputs, undesirable_output, weight, names, disp):
    solve = DEAProblem(inputs, outputs, undesirable_output, weight,disp).solve()
    status = pd.DataFrame.from_dict(solve[0], orient="index", columns=["status"])
    efficiency = pd.DataFrame.from_dict(
        solve[1], orient="index", columns=["efficiency"]
    )
    weight = pd.DataFrame.from_dict(solve[2], orient="index")
    results = pd.concat([names, status, efficiency, weight], axis=1)
    return results.round(decimals=4)

dir_data = r"E:\tencent files\chrome Download\Research\DEA\DEA_transport\Data_input"
xls = r"DEA_inout.xlsx"

os.chdir(r"E:\tencent files\chrome Download\Research\DEA\DEA_transport\Data_input\eff_strong_reg")

def results_output_excel(iteration):

    results = results_df(
        inout_data(iteration, input_columns),
        inout_data(iteration, output_columns),
        inout_data(iteration, undesirable_outputs),
        weight,
        inout_data(iteration, names),
        disp="strong disposability"
    )
    with open(str(iteration)+'.pickle',"wb") as file:
        pickle.dump(results,file)
    print("iteration {} has finished".format(iteration))
    return results


input_columns = ["Labor", "Fixed investment", "FDP"]
output_columns = ["GDP"]
undesirable_outputs = ["GWP", "PMFP"]
weight = [0, 0, 1 / 3, 1 / 3, 1 / 6, 1 / 6]
names = ["City name", "year"]
iteration = 1000

os.path.dirname(os.path.abspath(__file__))

if __name__ ==  '__main__': 
    set_lowpriority()
    with Pool() as p:
        p.map(results_output_excel, range(iteration))