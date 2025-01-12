#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 17:46:04 2025

@author: lewisvaughan
"""

import pandas as pd
import numpy as np
import cvxpy as cp # used for convex optimisation problems, like this one


#-------------------------------------------------------------------------------------------------------
# INPUT DATA
generator_data = pd.read_csv("provided_material/generators.csv", index_col=0)
node_demands = pd.read_csv("provided_material/hourlydemandbynode.csv", index_col=0) # 168 hours by 20 demand nodes
shift_factor_matrix = pd.read_csv("provided_material/shiftfactormatrix.csv", index_col=0) # Have lINE_ID as row index of the dataframe
line_ratings = pd.read_csv("provided_material/lineratings.csv", index_col=0) # 532x1 vector
#-------------------------------------------------------------------------------------------------------


# EXTRACT REQUIRED INFO FROM INPUT DATA
gen_IDs = np.array( generator_data.loc[:, "NODE"].astype(int) )
gen_marginal_costs = np.array( generator_data.loc[:, "MC"] ) # (£/MWh) 
gen_capacities = np.array( generator_data.loc[:, "CAP"] ) # (MW)

demand_IDs = np.array(node_demands.columns.astype(int))
node_IDs = np.array(shift_factor_matrix.columns.astype(int))


# GENERATOR AND DEMAND CONTRIBUTIONS AS MATRICES
# Generator contributions
gen_indices = {g_ID: np.where(node_IDs == g_ID)[0][0] for g_ID in gen_IDs}  # indices of generators in node_IDs
gen_identity_mat = np.zeros((len(node_IDs), len(gen_IDs))) # (428,5)
gen_identity_mat[list(gen_indices.values()), np.arange(len(gen_IDs))] = 1

# Demand contributions
demand_indices = {d_ID: np.where(node_IDs == d_ID)[0][0] for d_ID in demand_IDs}  # indices of demands in node_IDs
demand_identity_mat = np.zeros((len(node_IDs), len(demand_IDs))) # (428,20)
demand_identity_mat[list(demand_indices.values()), np.arange(len(demand_IDs))] = 1

#-------------------------------------------------------------------------------------------------------


# DECISION VARIABLES: q_supply - the power supply from each of the 5 generators
q_supply = cp.Variable( (len(gen_IDs), len(node_demands)), nonneg=True) # (MW)

# OBJECTIVE FUNCTION - minimise total dispatch costs

# Elementwise multiplication of cost and power generation for generators for each hour - entries are cost of generator i at time t (i by t matrix)
# Then sum calculates the total cost by summing all entries - this is what needs to be minimised 
# Minimising each hour separately could result in unnecessary fluctuations in generator dispatch - may not meet operational or practical constraints.
objective = cp.Minimize( cp.sum(  cp.multiply(gen_marginal_costs[:, None], q_supply)  ) ) 


# CONSTRAINTS
constraints = []

# (1) - Total hourly generation must equal the total hourly demand (as assuming no power loss/ have a perfect network)
"""
for t in range(len(node_demands)):
    total_hourly_generation = cp.sum(q_supply[:, t])
    total_hourly_demand = cp.sum(node_demands.iloc[t,:]) # sum all the demands at hour t
    constraints.append(total_hourly_generation == total_hourly_demand) # this is an equality constraint
"""
constraints.append(cp.sum(q_supply, axis=0) == cp.sum(node_demands.values.T, axis=0)) # does same as commented out code 
    

# (2) - Line flow constraints (with maximal power ratings)
# Calculate net hourly injections using matrix multiplication
net_injections = gen_identity_mat @ q_supply - demand_identity_mat @ node_demands.values.T  # (428x24)

# Compute power flows for all hours using the shift factor matrix
PF = shift_factor_matrix.values @ net_injections

# Add line flow constraints for each line
num_lines = shift_factor_matrix.shape[0]
line_ratings_vector = line_ratings.values.flatten()  # Convert to 1D array
constraints += [PF <= line_ratings_vector[:, None], PF >= -line_ratings_vector[:, None]] # PF between max and min limit


# (3) - Generator capacity constraints
"""
for g in range(len(gen_capacities)):
    for t in range(len(node_demands)):
        constraints.append(q_supply[g,t] <= gen_capacities[g])
"""
# No need for the for loops above - broadcasting is being done behind the scenes - extends to each hour column 
constraints += [q_supply <= gen_capacities[:, None]] # 5x24 and 5x1 - hourly, check supply value is less than capacity
        

#-------------------------------------------------------------------------------------------------------
# SOLVE THE OPTIMISATION PROBLEM
problem = cp.Problem(objective, constraints)
problem.solve(
    verbose=True,
    solver=cp.CBC,
    max_iters=100000  # Optional: Set a maximum number of iterations if needed
)

# Convert optimal generator dispatch to a pandas DataFrame for better readability
q_supply_table = pd.DataFrame(q_supply.value, index=gen_IDs, columns=node_demands.index)  # 5 rows for generators by 168 columns for hours

# RESULTS
if problem.status == cp.OPTIMAL:
    print("OPTIMAL SOLUTION FOUND!")
    print(f"Dispatch costs (£): {problem.value:.2f}")
    
    print("Optimal generator dispatch (MW):")
    print(q_supply_table)
else:
    print("NO OPTIMAL SOLUTION FOUND!")




    