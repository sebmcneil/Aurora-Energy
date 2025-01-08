import pandas as pd
import numpy as np
import cvxpy as cp

# Load data
# Generators data
generators = pd.read_csv('generators.csv')  # Generator data
gen_costs = generators["MC"].values  # Marginal costs (£/MWh)
gen_caps = generators["CAP"].values  # Generator capacities (MW)
gen_nodes = generators["NODE"].values  # Nodes where generators are located

# Hourly demand data
hourly_demand = pd.read_csv('hourlydemandbynode.csv')  # Hourly demand at 20 nodes
demand_nodes = hourly_demand.columns[1:].astype(int).values  # Nodes with demand
hourly_demand = hourly_demand.iloc[:, 1:].to_numpy()  # Demand data (hours x nodes)
total_hours = hourly_demand.shape[0]

# Line ratings
line_ratings = pd.read_csv('lineratings.csv')  # Transmission line ratings
line_limits = line_ratings["RATING_MW"].replace("inf", np.inf).astype(float).values

# Shift factor matrix
shift_factors = pd.read_csv('shiftfactormatrix.csv', index_col=0)  # PTDF matrix
shift_factors = shift_factors.apply(pd.to_numeric, errors='coerce')  # Convert to numeric
shift_factors.columns = shift_factors.columns.astype(int)
shift_factors = shift_factors.loc[:, demand_nodes]  # Restrict to demand nodes

# Problem dimensions
num_generators = len(gen_costs)
num_lines = shift_factors.shape[0]
num_demand_nodes = len(demand_nodes)

# Decision variables
P = cp.Variable((num_generators, total_hours))  # Power generation at each generator (MW)

# Objective function: Minimize total generation cost
objective = cp.Minimize(cp.sum(cp.multiply(gen_costs[:, None], P)))

# Constraints
constraints = []

# Generator capacity constraints
for g in range(num_generators):
    constraints.append(P[g, :] >= 0)  # Non-negative generation
    constraints.append(P[g, :] <= gen_caps[g])  # Capacity limits

# Electricity demand balance constraints
for t in range(total_hours):
    for i, node in enumerate(demand_nodes):
        gen_at_node = [g for g, n in enumerate(gen_nodes) if n == node]
        # Add a constraint directly for each demand node
        constraints.append(cp.sum(P[gen_at_node, t]) >= hourly_demand[t, i])

# # Transmission line flow constraints
# for l in range(num_lines):
#     for t in range(total_hours):
#         # Calculate line flow using the shift factor matrix
#         line_flow = shift_factors.iloc[l, :] @ (
#             cp.hstack([cp.sum(P[gen_at_node, t]) for gen_at_node in [[g for g, n in enumerate(gen_nodes) if n == node] for node in demand_nodes]]) - hourly_demand[t, :]
#         )
#         constraints.append(line_flow <= line_limits[l])  # Upper limit
#         constraints.append(line_flow >= -line_limits[l])  # Lower limit

# Transmission line flow constraints
for l in range(num_lines):
    for t in range(total_hours):
        # Calculate line flow using the shift factor matrix
        line_flow = cp.sum(
            cp.multiply(
                shift_factors.iloc[l, :].values,  # Convert Series to NumPy array
                cp.hstack([cp.sum(P[gen_at_node, t]) for gen_at_node in [[g for g, n in enumerate(gen_nodes) if n == node] for node in demand_nodes]]) - hourly_demand[t, :]
            )
        )
        constraints.append(line_flow <= line_limits[l])  # Upper limit
        constraints.append(line_flow >= -line_limits[l])  # Lower limit


# Solve the optimization problem
problem = cp.Problem(objective, constraints)
result = problem.solve(solver=cp.SCS, eps=1e-6, verbose=True)


# Results
print("Optimal generation schedule (MW):")
print(P.value)
print("Optimal cost (£):", problem.value)

# Additional outputs
print("Total demand at each hour (MW):", np.sum(hourly_demand, axis=1))
print("Total generator capacities (MW):", np.sum(gen_caps))
print("Line limits (MW):", line_limits)
