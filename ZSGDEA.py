#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import os
import pandas as pd
import pickle
import pulp

get_ipython().run_line_magic('load_ext', 'nb_black')


# In[3]:


class ZSG_DEAProblem:
    def __init__(
        self,
        inputs,
        outputs,
        bad_outs,
        weight_vector=None,
        directional_factor=None,
        returns="CRS",
        in_weights=[0, None],
        out_weights=[0, None],
        badout_weights=[0, None],
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.bad_outs = bad_outs
        self.returns = returns
        self.weight_vector = (
            weight_vector  # weight vector in directional distance function
        )

        self.J, self.I = self.inputs.shape  # no of DMUs, inputs
        _, self.R = self.outputs.shape  # no of outputs
        _, self.S = self.bad_outs.shape  # no of bad outputs
        self._i = range(self.I)  # inputs
        self._r = range(self.R)  # outputs
        self._s = range(self.S)  # bad_output
        self._j = range(self.J)  # DMUs
        if directional_factor == None:
            self.gx = self.inputs
            self.gy = self.outputs
            self.gb = self.bad_outs
        else:
            self.gx = directional_factor[: self.I]
            self.gy = directional_factor[self.I : (self.I + self.J)]
            self.gy = directional_factor[(self.I + self.J) :]

        self._in_weights = in_weights  # input weight restrictions
        self._out_weights = out_weights  # output weight restrictions
        self._badout_weights = badout_weights  # bad output weight restrictions

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
        prob = pulp.LpProblem("".join(["DMU_", str(j0)]), pulp.LpMinimize)
        self.weights = pulp.LpVariable.dicts(
            "Weight", (self._j), lowBound=self._in_weights[0]
        )
        self.betax = pulp.LpVariable.dicts(
            "scalingFactor_x", (self._i), lowBound=0, upBound=1
        )

        self.betay = pulp.LpVariable.dicts("scalingFactor_y", (self._r), lowBound=0)

        self.betab = pulp.LpVariable.dicts(
            "scalingFactor_b",
            (self._s),
            lowBound=0,
        )

        # Set up objective function
        prob += pulp.lpSum([self.betab[s] for s in self._s])

        # Set up constraints
        for i in self._i:
            prob += (
                pulp.lpSum(
                    [(self.weights[j0] * self.inputs.values[j0][i]) for j0 in self._j]
                )
                <= self.inputs.values[j0][i]
            )
        for r in self._r:
            prob += (
                pulp.lpSum(
                    [(self.weights[j0] * self.outputs.values[j0][r]) for j0 in self._j]
                )
                >= self.outputs.values[j0][r]
            )

        for s in self._s:  # weak disposability
            prob += (
                pulp.lpSum(
                    [(self.weights[j0] * self.bad_outs.values[j0][s]) for j0 in self._j]
                )
                == self.betab[s] * self.bad_outs.values[j0][s]
            )
        # Set returns to scale
        if self.returns == "VRS":
            prob += sum([weight for weight in self.weights]) == 1

        return prob

    def solve(self):
        """
        Iterate over the dictionary of DMUs' problems, solve them, and collate
        the results into a pandas dataframe.
        """

        sol_status = {}
        sol_weights = {}
        sol_objective_function = {}

        for ind, problem in list(self.dmus.items()):
            problem.solve()
            sol_status[ind] = pulp.LpStatus[problem.status]
            sol_weights[ind] = {}
            for v in problem.variables():
                sol_weights[ind][v.name] = v.varValue
            sol_objective_function[ind] = pulp.value(problem.objective)
        return sol_status, sol_objective_function, sol_weights

    def adjust_und_output(self):
        """
        after solving all the classic DEA efficiency calculation,
        calculate each DMU's ZSG-DEA efficiency according to its
        DEA efficiency.
        """
        sol_status, sol_objective_function, sol_weights = self.solve()
        hri_eff_dict = {}
        adjust_quantity_dict = {}
        for i in list(self._j):
            eff_ls, ineff_ls, ineff_denom = [], [], []
            for j in range(self.bad_outs.shape[0]):

                if sol_objective_function[j] == 1:
                    efficient_DMU_b = sum(
                        [
                            self.bad_outs.values[j][s]
                            for s in range(self.bad_outs.shape[1])
                        ]
                    )
                    eff_ls.append(efficient_DMU_b)
                else:
                    inefficient_DMU_b = sum(
                        [
                            self.bad_outs.values[j][s]
                            for s in range(self.bad_outs.shape[1])
                        ]
                    )

                    inefficient_DMU_denominator = sol_objective_function[j] * sum(
                        [
                            self.bad_outs.values[j][s]
                            for s in range(self.bad_outs.shape[1])
                        ]
                    )
                    ineff_ls.append(inefficient_DMU_b)
                    ineff_denom.append(inefficient_DMU_denominator)

            eff_DMU_sum, ineff_DMU_sum, ineff_DMU_sumproduct = (
                sum(eff_ls),
                sum(ineff_ls),
                sum(ineff_denom),
            )
            hri_eff = (
                sol_objective_function[i] * sum([eff_DMU_sum, ineff_DMU_sum])
            ) / (
                sum([eff_DMU_sum, ineff_DMU_sumproduct])
            )  # 计算zsg-efficiency
            hri_eff_dict[i] = hri_eff

        status = pd.DataFrame.from_dict(sol_status, orient="index", columns=["status"])
        objective = pd.DataFrame.from_dict(
            sol_objective_function, orient="index", columns=["objective_function"]
        )
        hri_eff_df = pd.DataFrame.from_dict(
            hri_eff_dict, orient="index", columns=["hri_score"]
        )
        weight = pd.DataFrame.from_dict(sol_weights, orient="index")
        results = pd.concat([status, objective, hri_eff_df, weight], axis=1)
        return results


# In[4]:


def read_dataframe(path, file_lstm):
    os.chdir(path)
    with open(file_lstm, "rb") as file:
        data_df = pickle.load(file)
    return data_df


# In[5]:


def read_data(data, column_name, year):
    data_col = data.loc[:, column_name]
    data_col_year = data_col.loc[data_col.index.get_level_values(1) == year]
    return data_col_year


# In[16]:


path = r"D:\tencent files\chrome Download\Research\DEA\DEA_carbon market\Data"

file_lstm_strong = r"Data_lstm_dropout0.05.pickle"
results_file_strong = r"ZSGDEA_strong"

file_lstm_moderate = r"Data_lstm_dropout0.4.pickle"
results_file_moderate = r"ZSGDEA_moderate"

file_lstm_weak = r"Data_lstm_dropout0.8.pickle"
results_file_weak = r"ZSGDEA_weak"


# In[15]:


column_in = ["Population", "Fixed asset", "Energy consumption"]
column_out = ["GDP"]
column_undout = ["CO2 emisson"]


# In[14]:


def calc_eff(year, data):
    data_in = read_data(data, column_in, year)
    data_out = read_data(data, column_out, year)
    data_undout = read_data(data, column_undout, year)
    names = pd.DataFrame([i for i, _ in read_data(data, column_undout, year).index])
    results = (
        ZSG_DEAProblem(data_in, data_out, data_undout)
        .adjust_und_output()
        .round(decimals=4)
    )
    results = pd.concat([names, results], axis=1)
    return results


# In[18]:


def pickle_file(year, data, results_file_name, original_path=path):
    results = calc_eff(year, data)
    os.chdir(original_path)
    if not os.path.exists(results_file_name):
        os.mkdir(results_file_name)
    os.chdir(results_file_name)
    with open(str(year) + ".pickle", "wb") as file_name:
        pickle.dump(results, file_name)


# In[10]:


def main(path, file_lstm, results_file):
    years = range(1997, 2031)
    data = read_dataframe(path, file_lstm)
    for year in years:
        pickle_file(year, data, results_file)


# In[21]:


main(path, file_lstm_weak, results_file_weak)

