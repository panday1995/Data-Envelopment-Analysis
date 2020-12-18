#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pulp


# In[15]:


class ML_index:
    """
    Help on class DEAProblem

    ML_index(inputs, outputs, bad_outs, weight_vector, directional_factor=None, returns='CRS',
                 in_weights=[0, None], out_weights=[0, None],badout_weights=[0, None])

    DEAProblem solves DEA model using directional distance function.

    Parameters:
    inputs: input data, DataFrame data
    outputs: output data, DataFrame data
    bad_outs: undesirable output data, DataFrame data
    weight_vector: weights for individual inputs and outputs. List data
    """

    def __init__(
        self,
        inputs_1,
        outputs_1,
        bad_outs_1,
        inputs_2,
        outputs_2,
        bad_outs_2,
        weight_vector,
        directional_factor=None,
        returns="CRS",
        disp="weak disposability",
        in_weights=[0, None],
        out_weights=[0, None],
        badout_weights=[0, None],
    ):
        self.inputs_1 = inputs_1
        self.outputs_1 = outputs_1
        self.bad_outs_1 = bad_outs_1

        self.inputs_2 = inputs_2
        self.outputs_2 = outputs_2
        self.bad_outs_2 = bad_outs_2

        self.returns = returns
        self.weight_vector = (
            weight_vector  # weight vector in directional distance function
        )
        self.disp = disp

        self.J, self.I = self.inputs_1.shape  # no of DMUs, inputs
        _, self.R = self.outputs_1.shape  # no of outputs
        _, self.S = self.bad_outs_1.shape  # no of bad outputs
        self._i = range(self.I)  # inputs
        self._r = range(self.R)  # outputs
        self._s = range(self.S)  # bad_output
        self._j = range(self.J)  # DMUs
        if directional_factor == None:
            pass
        else:
            self.gx = directional_factor[: self.I]
            self.gy = directional_factor[self.I : (self.I + self.J)]
            self.gy = directional_factor[(self.I + self.J) :]

        self._in_weights = in_weights  # input weight restrictions
        self._out_weights = out_weights  # output weight restrictions
        self._badout_weights = badout_weights  # bad output weight restrictions

        # creates dictionary of pulp.LpProblem objects for the DMUs
        self.dmus = self.solve_problems()

    def solve_problems(self):
        """
        Iterate over the DMU and create a dictionary of LP problems, one
        for each DMU.
        """

        dmu_dict_ddf11 = {}
        dmu_dict_ddf22 = {}
        dmu_dict_ddf12 = (
            {}
        )  # DDF of technology in Period 1 using Period 2 as Reference technology
        dmu_dict_ddf21 = (
            {}
        )  # DDF of technology in Period 2 using Period 1 as Reference technology
        dmu_dict_MI = {}  # MI_index
        dmu_dict_EC = {}  # efficiency_change
        dmu_dict_TC = {}  # technology_change
        for j0 in self._j:
            dmu_dict_ddf11[j0] = self._make_problem(
                j0,
                self.inputs_1,
                self.outputs_1,
                self.bad_outs_1,
                self.inputs_1,
                self.outputs_1,
                self.bad_outs_1,
            )
            dmu_dict_ddf22[j0] = self._make_problem(
                j0,
                self.inputs_2,
                self.outputs_2,
                self.bad_outs_2,
                self.inputs_2,
                self.outputs_2,
                self.bad_outs_2,
            )
            dmu_dict_ddf12[j0] = self._make_problem(
                j0,
                self.inputs_2,
                self.outputs_2,
                self.bad_outs_2,
                self.inputs_1,
                self.outputs_1,
                self.bad_outs_1,
            )
            dmu_dict_ddf21[j0] = self._make_problem(
                j0,
                self.inputs_1,
                self.outputs_1,
                self.bad_outs_1,
                self.inputs_2,
                self.outputs_2,
                self.bad_outs_2,
            )

            # ML_index calculation
            numerator = (1 + dmu_dict_ddf12[j0]) * (1 + dmu_dict_ddf11[j0])
            denominator = (1 + dmu_dict_ddf22[j0]) * (1 + dmu_dict_ddf21[j0])
            dmu_dict_MI[j0] = (numerator / denominator) ** (1 / 2)

            # Efficiency change calculation
            dmu_dict_EC[j0] = (1 + dmu_dict_ddf11[j0]) / (1 + dmu_dict_ddf22[j0])            
            # Technological change calculation
            numerator_TC = (1 + dmu_dict_ddf12[j0]) * (1 + dmu_dict_ddf22[j0])
            denominator_TC = (1 + dmu_dict_ddf11[j0]) * (1 + dmu_dict_ddf21[j0])
            dmu_dict_TC[j0] = (numerator_TC / denominator_TC)**(1/2)



        return dmu_dict_MI, dmu_dict_EC, dmu_dict_TC

    def _make_problem(
        self, j0, inputs1, outputs1, bad_outs1, inputs2, outputs2, bad_outs2
    ):
        """
        Create a pulp.LpProblem for a DMU.
        """
        # Set up pulp
        prob = pulp.LpProblem("".join(["DMU_", str(j0)]), pulp.LpMaximize)
        self.weights = pulp.LpVariable.dicts(
            "Weight", (self._j), lowBound=self._in_weights[0]
        )
        self.betax = pulp.LpVariable.dicts(
            "scalingFactor_x", (self._i), lowBound=0, upBound=1
        )

        self.betay = pulp.LpVariable.dicts("scalingFactor_y", (self._r), lowBound=0)

        self.betab = pulp.LpVariable.dicts(
            "scalingFactor_b", (self._s), lowBound=0, upBound=1
        )

        # Set up objective function
        prob += pulp.lpSum(
            [(self.weight_vector[i] * self.betax[i]) for i in self._i]
            + [(self.weight_vector[self.I + r] * self.betay[r]) for r in self._r]
            + [
                (self.weight_vector[self.I + self.R + s] * self.betab[s])
                for s in self._s
            ]
        )

        # Set up constraints
        for i in self._i:
            prob += (
                pulp.lpSum(
                    [(self.weights[j0] * inputs1.values[j0][i]) for j0 in self._j]
                )
                <= inputs2.values[j0][i] - self.betax[i] * inputs2.values[j0][i]
            )
        for r in self._r:
            prob += (
                pulp.lpSum(
                    [(self.weights[j0] * outputs1.values[j0][r]) for j0 in self._j]
                )
                >= outputs2.values[j0][r] + self.betay[r] * outputs2.values[j0][r]
            )

        if self.disp == "weak disposability":
            for s in self._s:  # weak disposability
                prob += (
                    pulp.lpSum(
                        [(self.weights[j0] * bad_outs1.values[j0][s]) for j0 in self._j]
                    )
                    == bad_outs2.values[j0][s] - self.betab[s] * bad_outs2.values[j0][s]
                )

        elif self.disp == "strong disposability":
            for s in self._s:  # strong disposability
                prob += (
                    pulp.lpSum(
                        [(self.weights[j0] * bad_outs1.values[j0][s]) for j0 in self._j]
                    )
                    >= bad_outs2.values[j0][s] - self.betab[s] * bad_outs2.values[j0][s]
                )

        # Set returns to scale
        if self.returns == "VRS":
            prob += sum([self.weights[j] for j in self.weights]) == 1

        prob.solve()
        Objective_func = pulp.value(prob.objective)
        return Objective_func


