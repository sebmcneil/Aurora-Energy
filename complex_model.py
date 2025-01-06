import pandas as pd
import numpy as np
import cvxpy as cp

# Load CSV files
generators = pd.read_csv('generators.csv')  # Generator data
hourly_demand = pd.read_csv('hourlydemandbynode.csv')  # Hourly demand
line_ratings = pd.read_csv('lineratings.csv')  # Line ratings
shift_factor_matrix = pd.read_csv('shiftfactormatrix.csv')  # Shift factor matrix

# Extract relevant data from `generators.csv`
gen_costs = generators["MC"].values  # Marginal costs (£/MWh)
gen_caps = generators["CAP"].values  # Generator capacities (MW)
num_generators = len(gen_costs)

# Generator-to-node mapping
nodes = list(shift_factor_matrix.columns[1:])  # Node IDs from the shift factor matrix
generator_nodes = generators["NODE"].astype(str).values
generator_to_node_matrix = np.zeros((len(nodes), num_generators))

for g, gen_node in enumerate(generator_nodes):
    if gen_node in nodes:
        generator_to_node_matrix[nodes.index(gen_node), g] = 1

# Process demand matrix
hourly_demand_nodes = hourly_demand.columns[1:].astype(str)
num_hours = len(hourly_demand)
num_nodes = len(nodes)

# Expand the demand matrix to include all 428 nodes
full_node_demand = np.zeros((num_hours, num_nodes))
node_indices_in_full_set = [nodes.index(node) for node in hourly_demand_nodes]
for i, idx in enumerate(node_indices_in_full_set):
    full_node_demand[:, idx] = hourly_demand.iloc[:, i + 1]

# Extract line limits from `line_ratings.csv`
line_limits = line_ratings["RATING_MW"].replace('inf', np.inf).astype(float).values
num_lines = len(line_limits)

# Extract the shift factors matrix
shift_factors = shift_factor_matrix.iloc[:, 1:].to_numpy()

# Ensure dimensions match
assert shift_factors.shape == (num_lines, num_nodes), "Shift factor matrix dimensions do not match!"

# Decision variables
P = cp.Variable((num_generators, num_hours))  # Power generation (MW)

# Objective function: Minimise total generation cost
objective = cp.Minimize(cp.sum(cp.multiply(gen_costs[:, None], P)))

# Constraints
constraints = []

# Generator capacity constraints
for g in range(num_generators):
    constraints.append(P[g, :] >= 0)  # Non-negative generation
    constraints.append(P[g, :] <= gen_caps[g])  # Capacity limits

# Electricity demand balance constraints and line flow constraints
for t in range(num_hours):
    # Net injection per node
    node_net_injections = generator_to_node_matrix @ P[:, t] - full_node_demand[t, :]
    
    for l in range(num_lines):
        line_flow = cp.sum(cp.multiply(shift_factors[l, :], node_net_injections))  # Line flow
        constraints.append(line_flow <= line_limits[l])  # Upper limit
        constraints.append(line_flow >= -line_limits[l])  # Lower limit

# Solve the optimisation problem
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.SCS)

# Results
optimal_generation = P.value
optimal_cost = problem.value

# Output results
print("Optimal generation schedule:")
print(optimal_generation)
print("Optimal cost (£):", optimal_cost)
