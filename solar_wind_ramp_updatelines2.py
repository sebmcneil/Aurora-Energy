
# SOLVER WITH WIND, SOLAR AND RAMPING CONSTRAINTS

import pandas as pd
import numpy as np
import cvxpy as cp
from ddeint import ddeint
import matplotlib.pyplot as plt


#  ALL FUNCTIONS DEFINED FIRST
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

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

    # Define the DDE model
    def model(P, t):
        return (P_target - P(t - delay)) / tau

    # Define the history function
    def history(t):
        return P_initial

    # Solve the DDE
    response = ddeint(model, history, time)

    # Approximate ramp rate as the slope over the simulation period
    ramp_rate = (response[-1] - response[0]) / duration
    return abs(ramp_rate)  # Return the absolute value of ramp rate
# -------------------------------------------------------------------------------------------------------


# Function for carrying out the optimisation
def run_optimization(line_ratings_vector):
    # DECISION VARIABLES: q_supply - power supply from each generator for each hour (5x168)
    q_supply = cp.Variable((len(gen_IDs), len(node_demands)), nonneg=True)  # (MW)

    # OBJECTIVE FUNCTION - minimise total dispatch costs
    objective = cp.Minimize(cp.sum(cp.multiply(gen_marginal_costs[:, None], q_supply)))

    # CONSTRAINTS
    constraints = []

    # (1) - Total hourly generation must equal the total hourly demand
    constraints.append(cp.sum(q_supply, axis=0) == cp.sum(node_demands.values.T, axis=0))

    # (2) - Line flow constraints
    net_injections = gen_identity_mat @ q_supply - demand_identity_mat @ node_demands.values.T
    PF = shift_factor_matrix.values @ net_injections

    finite_mask = np.isfinite(line_ratings.values.flatten())
    constraints += [
        PF[finite_mask] <= line_ratings.values.flatten()[finite_mask][:, None],
        PF[finite_mask] >= -line_ratings.values.flatten()[finite_mask][:, None],
    ]

    # (3) - Generator capacity constraints with hourly availability
    constraints += [q_supply <= gen_capacities[:, None] * availability_matrix]

    # (4) - Ensure remaining demand is met by Generators 3, 4, and 5
    remaining_demand = cp.sum(node_demands.values.T, axis=0) - cp.sum(q_supply[:2, :], axis=0)
    constraints.append(cp.sum(q_supply[2:, :], axis=0) >= remaining_demand)

    # (5) - Capacity constraints for Generators 3, 4, and 5
    constraints += [q_supply[2:, :] <= gen_capacities[2:, None]]

    
    # (6) - Ramping constraints for Generators 3, 4, and 5
    for t in range(1, hours):
        for g, ramp_rate in ramp_rates.items():
            gen_idx = g - 1  # Adjust for zero-based indexing
            constraints.append(q_supply[gen_idx, t] - q_supply[gen_idx, t - 1] <= ramp_rate)
            constraints.append(q_supply[gen_idx, t - 1] - q_supply[gen_idx, t] <= ramp_rate)
    
    # SOLVE THE OPTIMISATION PROBLEM
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=True, solver=cp.CBC)
    
    # results
    if problem.status == cp.OPTIMAL:
        q_supply_table = pd.DataFrame(q_supply.value, index=gen_IDs, columns=node_demands.index)
        # print("Optimal generator dispatch (MW):")
        # print(q_supply_table)
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
def resolve_bottlenecks(line_ratings_vector, max_iterations=5):
    """
    Resolves bottlenecks in a network by iteratively optimizing line ratings.
    
    Parameters:
        line_ratings_vector (np.array): Initial line ratings.
        max_iterations (int): Maximum number of iterations to attempt optimization.
    
    Returns:
        tuple or None: (dispatch_cost, q_supply_table, PF_values) if successful; None if no solution is found.
    """
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Step 1: Run optimization
        try:
            dispatch_cost, q_supply_table, PF_values = run_optimization(line_ratings_vector)
        except RuntimeError as e:
            print(f"Optimization failed: {e}")
            print("Optimization failed or no valid solution found")
            return None, None, None 
        
        # Step 2: Calculate utilization and identify bottlenecks
        _, bottleneck_lines = calculate_utilization(PF_values, line_ratings_vector)
                
        # If no bottlenecks, optimization is complete
        if bottleneck_lines.empty:
            print("\nNo bottlenecks detected. Optimization complete.")
            print(f"Final Dispatch Cost: £{dispatch_cost:.2f}")
            print("Optimal generator dispatch (MW):")
            print(q_supply_table)
            return dispatch_cost, q_supply_table, PF_values
        
        # Step 3: Adjust line ratings for bottleneck lines
        line_ratings_vector = adjust_line_ratings(line_ratings_vector, bottleneck_lines)
        
        # Print updated line ratings for bottleneck lines
        print("\nUpdated line ratings for bottleneck lines:")
        for line_id in bottleneck_lines.index:
            idx = np.where(shift_factor_matrix.index == line_id)[0][0]  # find index of each bottleneck line
            print(f"Line {line_id}: New Rating = {line_ratings_vector[idx]:.2f}")
    
    # If max iterations are reached without resolving bottlenecks
    print("\nMax iterations reached. Returning most recent solution with bottlenecks.")
    print(f"Final Dispatch Cost: £{dispatch_cost:.2f}")
    print("Optimal generator dispatch (MW):")
    print(q_supply_table)
    
    return dispatch_cost, q_supply_table, PF_values
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------


# START OF MAIN CODE BLOCK

# INPUT DATA
generator_data = pd.read_csv("provided_material/generators.csv", index_col=0)
node_demands = pd.read_csv("provided_material/hourlydemandbynode.csv", index_col=0) # 168 hours by 20 demand nodes
shift_factor_matrix = pd.read_csv("provided_material/shiftfactormatrix.csv", index_col=0) # Have lINE_ID as row index of the dataframe
line_ratings = pd.read_csv("provided_material/lineratings.csv", index_col=0) # 532x1 vector


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


# HOURLY AVAILABILITY FOR GENERATORS 1 AND 2
# Load solar irradiance data (output from NASA POWER API processing)
solar_data = pd.read_csv("solar_data_final.csv")
solar_availability = solar_data["Availability"].values  # Use raw availability values from the data

# Generate Weibull-distributed wind availability
np.random.seed(42)  # Replace 42 with any integer of your choice
hours = node_demands.shape[0]
shape, scale = 2.0, 0.8  # Example Weibull parameters
wind_availability = np.random.weibull(shape, hours) * scale
wind_availability = np.clip(wind_availability, 0.4, 1.0)  # Ensure at least 40% availability

# Create the availability matrix
availability_matrix = np.ones((len(gen_IDs), hours))
availability_matrix[0, :] = wind_availability  # Generator 1 (solar)
availability_matrix[1, :] = wind_availability  # Generator 2 (wind)


# DETERMINE RAMP RATES FOR GENERATORS 3, 4 AND 5
ramp_rates = {}
for gen_id, tau, delay, P_initial, P_target in zip(
    [3, 4, 5], [1.6, 1.7, 2.5], [0.55, 0.6, 1.0], [0, 0, 0], [gen_capacities[2], gen_capacities[3], gen_capacities[4]]
):
    ramp_rates[gen_id] = simulate_dde(tau, delay, P_initial, P_target)
print("Ramp rates (MW/hour):", ramp_rates)


line_ratings_vector = line_ratings.values.flatten()
dispatch_cost, q_supply_table, PF_values = resolve_bottlenecks(line_ratings_vector)

if dispatch_cost is not None:
    # PLOT THE DISPATCH FROM GENERATORS OVER THE WEEK
    plt.figure(figsize=(12, 6))
    for g in range(len(gen_IDs)):
        plt.plot(q_supply_table.iloc[g, :], label=f"Generator {g+1}")
    plt.xlabel("Hour")
    plt.ylabel("Power Output (MW)")
    plt.title("Generator Dispatch Over the Week")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# CAN USE THESE LINES OF CODE TO FIND OUT IF DEMAND > SUPPLY AT ANY HOUR - LEADS TO INFEASIBLE PROBLEM

total_demand_per_hour = np.sum(node_demands, axis=1).to_numpy(dtype=np.float64)
total_supply_per_hour = np.sum(availability_matrix * gen_capacities[:,None], axis=0)
hours_exceeding_demand = np.where(total_demand_per_hour > total_supply_per_hour)[0]
print("Hours where demand exceeds supply:", hours_exceeding_demand)



# All lines infinite - no ramping rates - no solar/wind distributions: £1,195,967.94
# Add line ratings - no ramping rates - no solar/wind distributions: £3,123,854.39

# Add line ratings - no ramping rates - add 2 wind generators: £13,704,939.64
# Add line ratings - no ramping rates - wind and solar generators for 1/2: - no solution found

# Add line ratings - add ramping rates - no solar/wind distributions: £3,412,577.17 -->  £3,412,577.17 on 2nd iteration too
# Add line ratings - add ramping rates - add 2 wind generators: £21,992,108.01 
# Add line ratings - add ramping rates - add wind and solar generators for 1/2: no solution found - hours [42 44 46] have demand>supply

 


