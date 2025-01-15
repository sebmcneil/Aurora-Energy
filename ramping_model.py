import pandas as pd
import numpy as np
import cvxpy as cp
from ddeint import ddeint
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------------------------
# INPUT DATA
generator_data = pd.read_csv("provided_material/generators.csv", index_col=0)
node_demands = pd.read_csv("provided_material/hourlydemandbynode.csv", index_col=0)  # 168 hours by 20 demand nodes
shift_factor_matrix = pd.read_csv("provided_material/shiftfactormatrix.csv", index_col=0)  # PTDF matrix
line_ratings = pd.read_csv("provided_material/lineratings.csv", index_col=0)  # Line ratings
#-------------------------------------------------------------------------------------------------------


# EXTRACT REQUIRED INFO FROM INPUT DATA
gen_IDs = np.array(generator_data.loc[:, "NODE"].astype(int))
gen_marginal_costs = np.array(generator_data.loc[:, "MC"])  # (£/MWh)
gen_capacities = np.array(generator_data.loc[:, "CAP"])  # (MW)

demand_IDs = np.array(node_demands.columns.astype(int))
node_IDs = np.array(shift_factor_matrix.columns.astype(int))

#-------------------------------------------------------------------------------------------------------
# FUNCTION TO SOLVE DDE AND DETERMINE RAMP RATES
def simulate_dde(tau, delay, P_initial, P_target, duration=10, steps=1000):
    """
    Simulates a generator's ramping response using a delay differential equation (DDE).
    Args:
        tau: Time constant for the generator.
        delay: Time delay before response begins.
        P_initial: Initial power output (MW).
        P_target: Target power output (MW).
        duration: Total simulation time (hours).
        steps: Number of time steps for simulation.
    Returns:
        ramp_rate: Approximate ramp rate (MW/hour).
    """
    time = np.linspace(0, duration, steps)
    
    def model(Y, t):
        return (P_target - Y(t - delay)) / tau
    
    history = lambda t: P_initial  # Initial condition
    response = ddeint(model, history, time)
    
    # Approximate ramp rate: Change in power over change in time
    ramp_rate = (response[-1] - response[0]) / duration
    return abs(ramp_rate)  # Return the absolute value of ramp rate

# Determine ramp rates for generators 3, 4, and 5
ramp_rates = {}
for gen_id, tau, delay, P_initial, P_target in zip(
    [3, 4, 5], [1.5, 2.0, 2.5], [0.5, 0.7, 1.0], [0, 0, 0], [gen_capacities[2], gen_capacities[3], gen_capacities[4]]
):
    ramp_rates[gen_id] = simulate_dde(tau, delay, P_initial, P_target)
print("Ramp rates (MW/hour):", ramp_rates)

#-------------------------------------------------------------------------------------------------------

# DECISION VARIABLES: q_supply - power supply from each generator for each hour (5x168)
q_supply = cp.Variable((len(gen_IDs), len(node_demands)), nonneg=True)  # (MW)

# OBJECTIVE FUNCTION - minimise total dispatch costs
objective = cp.Minimize(cp.sum(cp.multiply(gen_marginal_costs[:, None], q_supply)))

#-------------------------------------------------------------------------------------------------------

# CONSTRAINTS
constraints = []

# (1) - Total hourly generation must equal the total hourly demand
constraints.append(cp.sum(q_supply, axis=0) == cp.sum(node_demands.values.T, axis=0))

# (2) - Line flow constraints
gen_indices = {g_ID: np.where(node_IDs == g_ID)[0][0] for g_ID in gen_IDs}
gen_identity_mat = np.zeros((len(node_IDs), len(gen_IDs)))
gen_identity_mat[list(gen_indices.values()), np.arange(len(gen_IDs))] = 1

demand_indices = {d_ID: np.where(node_IDs == d_ID)[0][0] for d_ID in demand_IDs}
demand_identity_mat = np.zeros((len(node_IDs), len(demand_IDs)))
demand_identity_mat[list(demand_indices.values()), np.arange(len(demand_IDs))] = 1

net_injections = gen_identity_mat @ q_supply - demand_identity_mat @ node_demands.values.T
PF = shift_factor_matrix.values @ net_injections

line_ratings_vector = line_ratings.values.flatten()
finite_mask = np.isfinite(line_ratings_vector)

constraints += [
    PF[finite_mask] <= line_ratings_vector[finite_mask][:, None],
    PF[finite_mask] >= -line_ratings_vector[finite_mask][:, None],
]

# (3) - Generator capacity constraints
constraints += [q_supply <= gen_capacities[:, None]]

# (4) - Ramping constraints for generators 3, 4, and 5
for t in range(1, len(node_demands)):
    for g, ramp_rate in ramp_rates.items():
        gen_idx = g - 1  # Adjust for zero-based indexing
        constraints.append(q_supply[gen_idx, t] - q_supply[gen_idx, t - 1] <= ramp_rate)
        constraints.append(q_supply[gen_idx, t - 1] - q_supply[gen_idx, t] <= ramp_rate)

#-------------------------------------------------------------------------------------------------------

# SOLVE THE OPTIMISATION PROBLEM
problem = cp.Problem(objective, constraints)
problem.solve(verbose=True, solver=cp.CBC)

# RESULTS
if problem.status == cp.OPTIMAL:
    print("OPTIMAL SOLUTION FOUND!")
    print(f"Dispatch costs (£): {problem.value:.2f}")
else:
    print("NO OPTIMAL SOLUTION FOUND!")

q_supply_table = pd.DataFrame(q_supply.value, index=gen_IDs, columns=node_demands.index)
print("Optimal generator dispatch (MW):")
print(q_supply_table)

#-------------------------------------------------------------------------------------------------------
