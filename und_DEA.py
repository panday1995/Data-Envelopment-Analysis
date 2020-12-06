
import numpy as np
import os
import pandas as pd
import pickle
import pulp

class DEAProblem:
    def __init__(
        self,
        inputs,
        outputs,
        bad_outs,
        weight_vector,
        directional_factor=None,
        returns="CRS",
        disp = 'weak disposability',
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
        self.disp = disp

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

        # Set returns to scale
        if self.returns == "VRS":
            prob += pulp.lpSum([self.weights[j] for j in self.weights]) == 1

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
                    [(self.weights[j0] * self.inputs.values[j0][i]) for j0 in self._j]
                )
                <= self.inputs.values[j0][i] - self.betax[i] * self.gx.values[j0][i]
            )
        for r in self._r:
            prob += (
                pulp.lpSum(
                    [(self.weights[j0] * self.outputs.values[j0][r]) for j0 in self._j]
                )
                >= self.outputs.values[j0][r] + self.betay[r] * self.gy.values[j0][r]
            )

        if self.disp == "weak disposability":
            for s in self._s:  # weak disposability
                prob += (
                    pulp.lpSum(
                        [
                            (self.weights[j0] * self.bad_outs.values[j0][s])
                            for j0 in self._j
                        ]
                    )
                    == self.bad_outs.values[j0][s]
                    - self.betab[s] * self.gb.values[j0][s]
                )

        elif self.disp == "strong disposability":
            for s in self._s:  # strong disposability
                prob += (
                    pulp.lpSum(
                        [
                            (self.weights[j0] * self.bad_outs.values[j0][s])
                            for j0 in self._j
                        ]
                    )
                    >= self.bad_outs.values[j0][s]
                    - self.betab[s] * self.gb.values[j0][s]
                )

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
