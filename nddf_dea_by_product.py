# %%
import numpy as np
import pandas as pd
import pulp


# %%
class DEAProblem:
    """
    Help on class DEAProblem

    DEAProblem(inputs, outputs, bad_outs, weight_vector, directional_factor=None, returns='CRS',
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
        n_inputs,
        p_inputs,
        outputs,
        bad_outs,
        weight_vector,
        directional_factor=None,
        returns="CRS",
        in_weights=[0, None],
        out_weights=[0, None],
        badout_weights=[0, None],
    ):
        self.n_inputs = n_inputs
        self.p_inputs = p_inputs
        self.outputs = outputs
        self.bad_outs = bad_outs
        self.returns = returns
        self.weight_vector = (
            weight_vector  # weight vector in directional distance function
        )

        self.J, self.nI = self.n_inputs.shape  # no of DMUs, non-polluting inputs
        _, self.pI = self.p_inputs.shape  # no of polluting inputs
        _, self.R = self.outputs.shape  # no of outputs
        _, self.S = self.bad_outs.shape  # no of bad outputs
        self._ni = range(self.nI)  # iterate over non-polluting inputs
        self._pi = range(self.pI)  # iterable over polluting inputs
        self._r = range(self.R)  # outputs
        self._s = range(self.S)  # bad_output
        self._j = range(self.J)  # DMUs
        if directional_factor == None:
            self.gx1 = self.n_inputs
            self.gx2 = self.p_inputs
            self.gy = self.outputs
            self.gb = self.bad_outs
        else:
            self.gx1 = directional_factor[: self.nI]
            self.gx2 = directional_factor[: (self.nI + self.pI)]
            self.gy = directional_factor[
                (self.nI + self.pI) : ((self.nI + self.pI) + self.J)
            ]
            self.gy = directional_factor[((self.nI + self.pI) + self.J) :]

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
        Create a by-product technology model. Reference: doi.org/10.1111/deci.12421
        """
        # Set up pulp
        prob = pulp.LpProblem("".join(["DMU_", str(j0)]), pulp.LpMaximize)
        self.weights_1 = pulp.LpVariable.dicts(
            "Weight_non_polluting", (self._j), lowBound=self._in_weights[0]
        )  # define the weight for non-pulluting unit in the by-product technology model

        self.weights_2 = pulp.LpVariable.dicts(
            "Weight_polluting", (self._j), lowBound=self._in_weights[0]
        )  # define the weight for pulluting unit in the by-product technology model

        #         self.betax1 = pulp.LpVariable.dicts(
        #             "scalingFactor_nx", (self._ni), lowBound=0, upBound=1
        #         ) # scaling factor for non-polluting inputs
        #         self.betax2 = pulp.LpVariable.dicts(
        #             "scalingFactor_px", (self._pi), lowBound=0, upBound=1
        #         ) # scaling factor for polluting inputs

        self.beta1 = pulp.LpVariable(
            "scalingFactor_y", lowBound=0
        )  # scaling factor for desirable output
        self.beta2 = pulp.LpVariable(
            "scalingFactor_b", lowBound=0
        )  # scaling factor for desirable output

        #         self.betab = pulp.LpVariable.dicts(
        #             "scalingFactor_b", (self._s), lowBound=0, upBound=1
        #         ) # scaling factor for undesirable factor

        # Set up objective function
        prob += pulp.lpSum([self.beta1, self.beta2]) / 2

        # Set up constraints
        for ni in self._ni:
            prob += (
                pulp.lpSum(
                    [
                        (self.weights_1[j0] * self.n_inputs.values[j0][ni])
                        for j0 in self._j
                    ]
                )
                <= self.n_inputs.values[j0][ni]
            )
        for pi in self._pi:
            prob += (
                pulp.lpSum(
                    [
                        (self.weights_1[j0] * self.p_inputs.values[j0][pi])
                        for j0 in self._j
                    ]
                )
                <= self.p_inputs.values[j0][pi]
            )
        # strong disposability for desirable output in non-pulluting process
        for r in self._r:
            prob += (
                pulp.lpSum(
                    [
                        (self.weights_1[j0] * self.outputs.values[j0][r])
                        for j0 in self._j
                    ]
                )
                >= self.outputs.values[j0][r] + self.beta1 * self.gy.values[j0][r]
            )

        for pi in self._pi:
            prob += (
                pulp.lpSum(
                    [
                        (self.weights_2[j0] * self.p_inputs.values[j0][pi])
                        for j0 in self._j
                    ]
                )
                >= self.p_inputs.values[j0][pi]
            )
        # strong disposability for undesirable output in polluting process
        for s in self._s:
            prob += (
                pulp.lpSum(
                    [
                        (self.weights_2[j0] * self.bad_outs.values[j0][s])
                        for j0 in self._j
                    ]
                )
                <= self.bad_outs.values[j0][s] - self.beta2 * self.gb.values[j0][s]
            )

        # Set returns to scale
        if self.returns == "VRS":
            prob += sum([self.weights_1[j] for j in self.weights_1]) == 1
            prob += sum([self.weights_2[j] for j in self.weights_2]) == 1

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
            problem.solve(pulp.PULP_CBC_CMD(msg=1))  #
            sol_status[ind] = pulp.LpStatus[problem.status]
            sol_weights[ind] = {}
            for v in problem.variables():
                sol_weights[ind][v.name] = v.varValue

            sol_efficiency[ind] = pulp.value(problem.objective)
        #             for name, c in list(problem.constraints.items()):
        #                 print(name, ":", c, "\t", c.pi, "\t\t", c.slack)
        return sol_status, sol_efficiency, sol_weights


# %%
n_X = pd.DataFrame(
    np.array(
        [
            [20],
            [30],
            [40],
            [20],
            [10],
            [11],
            [12],
            [14],
        ]
    )
)
p_X = pd.DataFrame(
    np.array(
        [
            [300],
            [200],
            [100],
            [200],
            [400],
            [222],
            [321],
            [231],
        ]
    )
)
y = pd.DataFrame(np.array([[20], [30], [40], [30], [50], [21], [32], [42]]))
b = pd.DataFrame(np.array([[10], [20], [10], [10], [10], [12], [11], [10]]))
weight = [0, 0, 1 / 2, 1 / 2]
names = pd.DataFrame(
    ["Bratislava", "Zilina", "Kosice", "Presov", "Poprad", "ala", "ba", "ca"],
    columns=["DMU"],
)

# %%
solve = DEAProblem(n_X, p_X, y, b, weight).solve()

# %%
status = pd.DataFrame.from_dict(solve[0], orient="index", columns=["status"])
efficiency = pd.DataFrame.from_dict(solve[1], orient="index", columns=["efficiency"])
weights = pd.DataFrame.from_dict(solve[2], orient="index")
results = pd.concat([names, status, efficiency, weights], axis=1)

# %%
print(results.round(decimals=4))
