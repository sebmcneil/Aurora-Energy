import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

###### following section loads the data from the csv files and extracts the variables and dimensions to solve the problem

# Load CSV files
generators = pd.read_csv('generators.csv')  # Generator data
hourly_demand = pd.read_csv('hourlydemandbynode.csv')  # Hourly demand at nodes


# Extract relevant data from `generators.csv`
gen_costs = generators["MC"].values  # Marginal costs (£/MWh)
gen_caps = generators["CAP"].values  # Generator capacities (MW)

# Extract demand data from `hourly_demandbynode.csv`
hourly_demand = hourly_demand.iloc[:, 1:]  # Exclude the first column (hours)
#print(hourly_demand.head())
#print(hourly_demand.iloc[:, 0])
#sum_of_col_1 = np.sum(hourly_demand.iloc[:, 0])
#print(sum_of_col_1)
total_hourly_demand = np.sum(hourly_demand, axis=1)  # Total demand across all nodes for each hour
#print(total_hourly_demand)

# Problem dimensions
num_generators = len(gen_costs)
num_hours = len(total_hourly_demand)

######

###### this section defines the decision variables and objectives for the problem

# Decision variables
## this sets up the power output of each generator for each hour as decision variables
P = cp.Variable((num_generators, num_hours))  # Power generation (MW)

# Objective function: Minimise total generation cost
## defines the objective to minimise cost
objective = cp.Minimize(cp.sum(cp.multiply(gen_costs[:, None], P)))

######

###### this section adds the constraints (generator capacity limits and ensures demand is met)

# Constraints
constraints = []

# Generator capacity constraints
for g in range(num_generators):
    constraints.append(P[g, :] >= 0)  # Non-negative generation
    constraints.append(P[g, :] <= gen_caps[g])  # Capacity limits

# Electricity demand balance constraints
for t in range(num_hours):
    constraints.append(cp.sum(P[:, t]) == total_hourly_demand[t])  # Total generation = Total demand

######

###### this section solves the problem using SCS

# Solve the optimisation problem
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.SCS)  # Use SCS solver

# Results
optimal_generation = P.value
optimal_cost = problem.value

# Print results
print("Optimal generation: ", optimal_generation) 
print("Optimal cost: ",optimal_cost)

######

###### plots

plt.figure(figsize=(12, 6))
for g in range(num_generators):
    plt.plot(optimal_generation[g, :], label=f"Generator {g+1}")
plt.plot(total_hourly_demand, label="Total Demand (MW)", color='black', linestyle='--', linewidth=2)
plt.xlabel("Hour")
plt.ylabel("Power (MW)")
plt.title("Generation Contribution by Generator")
plt.legend()
plt.show()

total_cost_by_gen = np.sum(optimal_generation * gen_costs[:, None], axis=1)

plt.figure(figsize=(6, 6))
plt.pie(total_cost_by_gen, labels=[f"Generator {i+1}" for i in range(num_generators)], autopct='%1.1f%%')
plt.title("Cost Contribution by Generator")
plt.show()


