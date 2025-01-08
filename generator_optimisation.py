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
# node_demands = node_demands.iloc[:24, :] # to look at first day only

shift_factor_matrix = pd.read_csv("provided_material/shiftfactormatrix.csv", index_col=0) # Have lINE_ID as row index of the dataframe
line_ratings = pd.read_csv("provided_material/lineratings.csv", index_col=0) # 532x1 vector
#-------------------------------------------------------------------------------------------------------


# EXTRACT REQUIRED INFO FROM INPUT DATA
gen_IDs = np.array( generator_data.loc[:, "NODE"].astype(int) )
gen_marginal_costs = np.array( generator_data.loc[:, "MC"] ) # (£/MWh) 
gen_capacities = np.array( generator_data.loc[:, "CAP"] ) # (MW)

demand_IDs = np.array(node_demands.columns.astype(int))
node_IDs = np.array(shift_factor_matrix.columns.astype(int))



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
for t in range(len(node_demands)):
    total_hourly_generation = cp.sum(q_supply[:, t])
    total_hourly_demand = cp.sum(node_demands.iloc[t,:]) # sum all the demands at hour t
    constraints.append(total_hourly_generation == total_hourly_demand) # this is an equality constraint


# (2) - Line flow constraints (with maximal power ratings)
gen_indices = {g_ID: np.where(node_IDs == g_ID)[0][0] for g_ID in gen_IDs}  # indices of the generators in list of node_IDs
demand_indices = {d_ID: np.where(node_IDs == d_ID)[0][0] for d_ID in demand_IDs} # indices of demand nodes in list of node_IDs

for t in range(len(node_demands)):
    # Initialize net hourly injection symbolically - so can use q_supply (variable solving for) in calculations
    net_hourly_injection = cp.Variable(len(node_IDs))  # symbolic net injections at time t

    # Add generation contributions to net injections
    for i, g_ID in enumerate(gen_IDs):
        g_idx = gen_indices[g_ID]
        net_hourly_injection += cp.Constant(np.eye(len(node_IDs))[:, g_idx]) * q_supply[i, t]

    # Subtract demand contributions from net injections
    for i, d_ID in enumerate(demand_IDs):
        d_idx = demand_indices[d_ID] 
        net_hourly_injection -= cp.Constant(np.eye(len(node_IDs))[:, d_idx]) * node_demands.iloc[t, i]
        
    # Compute power flows symbolically
    PF = shift_factor_matrix.values @ net_hourly_injection  # matrix-vector multiplication

    # Add line flow constraints
    num_lines = shift_factor_matrix.shape[0]  # number of rows in the shift factor matrix
    for line_idx in range(num_lines):
        line_rating = line_ratings.iloc[line_idx, 0]  # line rating for the current line

        if line_rating != float('inf'):  # Add constraint only if line rating is finite
            constraints.append(PF[line_idx] <= line_rating)  # Upper bound
            constraints.append(PF[line_idx] >= -line_rating)  # Lower bound


# (3) - Generator capacity constraints
for g in range(len(gen_capacities)):
    for t in range(len(node_demands)):
        constraints.append(q_supply[g,t] <= gen_capacities[g])
        
        
#-------------------------------------------------------------------------------------------------------
# SOLVE THE OPTIMISATION PROBLEM
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.OSQP, verbose=True) # verbose to show progress of optimisation within the python console

# Convert optimal generator dispatch to a pandas DataFrame for better readability
q_supply_table = pd.DataFrame(q_supply.value, index=gen_IDs, columns=node_demands.index) # 5 rows for generators by 168 columns for hours

# RESULTS
if problem.status == cp.OPTIMAL:
    print("OPTIMAL SOLUTION FOUND!")
    print(f"Dispatch costs (£): {problem.value:.2f}")
    
    print("Optimal generator dispatch (MW):")
    print(q_supply_table)
else:
    print("NO OPTIMAL SOLUTION FOUND!")
    