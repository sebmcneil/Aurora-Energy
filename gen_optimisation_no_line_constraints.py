#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:37:27 2024

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


# (2) - Line flow constraints (no limits, only calculate the flows)
gen_indices = {g_ID: np.where(node_IDs == g_ID)[0][0] for g_ID in gen_IDs} # Indices of the generators in list of node_IDs
demand_indices = {d_ID: np.where(node_IDs == d_ID)[0][0] for d_ID in demand_IDs} # Indices of demand nodes in list of node_IDs

# Compute the net injections for all nodes 
for t in range(len(node_demands)):
    net_hourly_injection = np.zeros(len(node_IDs))
    
    # Assign generation to generator nodes using the precomputed indices
    for i, g_ID in enumerate(gen_IDs):  # iterating over generator IDs
        g_idx = gen_indices[g_ID]  # use precomputed index
        net_hourly_injection[g_idx] = q_supply[i, t].value  # generation at time t
    
    # Subtract demand contributions from net injections
    for i, d_ID in enumerate(demand_IDs):
        d_idx = demand_indices[d_ID] 
        net_hourly_injection[d_idx] -= node_demands.iloc[t,i] 
    
    # Find the power flows (note no constraints on this PF until transmission limits are included)
    PF = shift_factor_matrix @ net_hourly_injection  # matrix multiplication: (532x428) x (428x1) = (532x1) - the PF in each line


# (3) - Generator capacity constraints
for g in range(len(gen_capacities)):
    for t in range(len(node_demands)):
        constraints.append(q_supply[g,t] <= gen_capacities[g])
        
#-------------------------------------------------------------------------------------------------------  
# SOLVE THE OPTIMISATION PROBLEM
problem = cp.Problem(objective, constraints)
problem.solve()

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
        
    
    
    
    
    
    
    
    
    
    