#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:54:59 2025

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
gen_identity_mat[list(gen_indices.values()), np.arange(len(gen_IDs))] = 1 # Mark each generator for each column as a 1

# Demand contributions
demand_indices = {d_ID: np.where(node_IDs == d_ID)[0][0] for d_ID in demand_IDs}  # indices of demands in node_IDs
demand_identity_mat = np.zeros((len(node_IDs), len(demand_IDs))) # (428,20)
demand_identity_mat[list(demand_indices.values()), np.arange(len(demand_IDs))] = 1 # Mark each demand node for each column as a 1
#-------------------------------------------------------------------------------------------------------


# Function for carrying out the optimisation
def run_optimization(line_ratings_vector):
    # DECISION VARIABLES: q_supply - the power supply from each of the 5 generators for each hour (5x168)
    q_supply = cp.Variable( (len(gen_IDs), len(node_demands)), nonneg=True) # (MW)

    # OBJECTIVE FUNCTION - minimise total dispatch costs
    # Elementwise multiplication of cost and power generation for generators for each hour - entries are cost of generator i at time t (i by t matrix)
    # Then sum calculates the total cost by summing all entries - this is what needs to be minimised 
    # Minimising each hour separately could result in unnecessary fluctuations in generator dispatch - may not meet operational or practical constraints.
    objective = cp.Minimize( cp.sum(cp.multiply(gen_marginal_costs[:, None], q_supply))) 

    # CONSTRAINTS
    constraints = []

    # (1) - Total hourly generation must equal the total hourly demand (as assuming no power loss/ have a perfect network)
    constraints.append(cp.sum(q_supply, axis=0) == cp.sum(node_demands.values.T, axis=0)) # axis=0 goes through columns (each hour)
        
    # (2) - Line flow constraints (with maximal power ratings)
    # Calculate net hourly injections using matrix multiplication. Note negative values for demand rather than generation
    net_injections = gen_identity_mat @ q_supply - demand_identity_mat @ node_demands.values.T  # (428x168)

    # Compute power flows for all hours using the shift factor matrix
    PF = shift_factor_matrix.values @ net_injections

    finite_mask = np.isfinite(line_ratings_vector) # remove the 'inf' line ratings - only need lines with constraints
    constraints += [ # add the constraint that power in line is below its maximal rating 
        PF[finite_mask] <= line_ratings_vector[finite_mask][:, None],
        PF[finite_mask] >= -line_ratings_vector[finite_mask][:, None]
    ]

    # (3) - Generator capacity constraints
    constraints += [q_supply <= gen_capacities[:, None]] # 5x168 and 5x1 - hourly, check supply value is less than capacity
    # Note broadcasting is done to extend gen_capacities for 168 hours - allows for multiplication without using 2 for loops

        
    # SOLVE THE OPTIMISATION PROBLEM
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False, solver=cp.CBC)
    # This is a linear problem with linear constraints - CBC is quick and can be used
    # Other options like Clarabel, SCIPY, ECOS etc. are all compatible with this code if installed - they give same answer
    
    # results
    if problem.status == cp.OPTIMAL:
        q_supply_table = pd.DataFrame(q_supply.value, index=gen_IDs, columns=node_demands.index)
        return problem.value, q_supply_table, PF.value
    else:
        raise RuntimeError("No optimal solution found!")
#-------------------------------------------------------------------------------------------------------


# Function to calculate line utilization and identify bottlenecks
def calculate_utilization(PF_values, line_ratings_vector):
    finite_mask = np.isfinite(line_ratings_vector)  # consider only finite-rated lines
    
    # Calculate utilization (percentage of transmission line power limits)
    utilization = (np.abs(PF_values[finite_mask]) / line_ratings_vector[finite_mask][:, None] * 100)
    
    # Calculate max and average utilization per line
    max_utilization = np.max(utilization, axis=1)
    avg_utilization = np.mean(utilization, axis=1)
    
    # Create summary table
    line_analysis = pd.DataFrame({
        "Max Utilization (%)": max_utilization,
        "Avg Utilization (%)": avg_utilization,
    }, index=shift_factor_matrix.index[finite_mask])
    
    line_analysis = line_analysis.sort_values(by="Max Utilization (%)", ascending=False) # order table so bottlenecks at top
    
    # Identify bottleneck lines (max utilization >= 100%)
    bottleneck_lines = line_analysis[line_analysis["Max Utilization (%)"] >= 100]
    
    print("\nLine Utilization Summary:")
    print(line_analysis)
    print("\nBottleneck Lines:")
    print(bottleneck_lines)
    
    return line_analysis, bottleneck_lines
#-------------------------------------------------------------------------------------------------------


# Function to adjust line ratings for bottleneck lines
def adjust_line_ratings(line_ratings_vector, bottleneck_lines):
    new_ratings = line_ratings_vector.copy()
    for line_id in bottleneck_lines.index:
        idx = np.where(shift_factor_matrix.index == line_id)[0][0]  # find index of each bottleneck line
        new_ratings[idx] *= 1.10  # increase limit by 10%
        
    return new_ratings
#-------------------------------------------------------------------------------------------------------


# Main workflow
def resolve_bottlenecks(line_ratings_vector, max_iterations=4):
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Step 1: Run optimization
        try:
            dispatch_cost, q_supply_table, PF_values = run_optimization(line_ratings_vector)
        except RuntimeError as e:
            print(f"Optimization failed: {e}")
            break
        
        # Step 2: Calculate utilization and identify bottlenecks
        _, bottleneck_lines = calculate_utilization(PF_values, line_ratings_vector)
                
        # If no bottlenecks, optimization is complete
        if bottleneck_lines.empty:
            print("\nNo bottlenecks detected. Optimization complete.")
            print(f"Final Dispatch Cost: £{dispatch_cost:.2f}") 
            return dispatch_cost, q_supply_table, PF_values
        
        # Step 3: Adjust line ratings for bottleneck lines
        line_ratings_vector = adjust_line_ratings(line_ratings_vector, bottleneck_lines)
        
        # Print updated line ratings for bottleneck lines
        print("\nUpdated line ratings for bottleneck lines:")
        for line_id in bottleneck_lines.index:
            idx = np.where(shift_factor_matrix.index == line_id)[0][0]  # find index of each bottleneck line
            print(f"Line {line_id}: New Rating = {line_ratings_vector[idx]:.2f}")
    
    print("\nMax iterations reached. Optimization ended.")
    print(f"Final Dispatch Cost: £{dispatch_cost:.2f}") 
    return dispatch_cost, q_supply_table, PF_values
#-------------------------------------------------------------------------------------------------------


dispatch_cost, q_supply_table, PF_values = resolve_bottlenecks(line_ratings.values.flatten())
"""
L430 is the only line that reached 100% of its transmission line rating. Increasing its maximal rating can avoid a bottleneck 
and dispatch cost can be reduced massively. Increase any bottlenecks by 10% for each run
L430 - when constrained to 200MW
L430 - when constrained to 220MW
L430 - when constrained to 242MW
No bottlenecks when constrained to 266.20

Disptch costs - £1195967.94

"""










