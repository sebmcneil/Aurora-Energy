import pandas as pd
import numpy as np
import cvxpy as cp

# Load CSV files
generators = pd.read_csv('generators.csv')  # Generator data
hourly_demand = pd.read_csv('hourlydemandbynode.csv')  # Hourly demand at nodes

# Extract relevant data from `generators.csv`
gen_costs = generators["MC"].values  # Marginal costs (£/MWh)
gen_caps = generators["CAP"].values  # Generator capacities (MW)

# Extract demand data from `hourly_demandbynode.csv`
hourly_demand = hourly_demand.iloc[:, 1:].to_numpy()  # Exclude the first column (hours)
total_hourly_demand = np.sum(hourly_demand, axis=1)  # Total demand across all nodes for each hour

# Problem dimensions
num_generators = len(gen_costs)
num_hours = len(total_hourly_demand)

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

# Electricity demand balance constraints
for t in range(num_hours):
    constraints.append(cp.sum(P[:, t]) == total_hourly_demand[t])  # Total generation = Total demand

# Solve the optimisation problem
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.SCS)  # Use SCS solver

# Results
optimal_generation = P.value
optimal_cost = problem.value

# Print results
print("Optimal generation: ", optimal_generation) 
print("Optimal cost: ",optimal_cost)
