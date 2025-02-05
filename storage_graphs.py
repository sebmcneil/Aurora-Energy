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

gen_capacities[3] *= 1.2  # Capacity of generator 4 goes to 120 from 100

gen_capacities[0] *= 2 # if want to double the wind generator

#gen_capacities[1] *= 2 # if want to double the solar generator

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

hours = node_demands.shape[0]

time = np.arange(hours)

# Solar availability as a sine wave (Generator 2)

solar_availability = 0.5 * (1 + np.sin(2 * np.pi * time / 24 - np.pi / 2))  # Sine wave shifted to make nighttime 0

solar_availability = np.clip(solar_availability, 0, 1)  # Ensure values are between 0 and 1

# Generate Weibull-distributed wind availability

np.random.seed(42)  # Replace 42 with any integer of your choice

shape, scale = 2.3, 9.5  # Example Weibull parameters

wind_availability_gen1 = np.random.weibull(shape, hours) * scale

wind_availability_gen2 = np.random.weibull(shape, hours) * scale

max_possible_speed = scale * 1.5  # Assume maximum wind speed reaches ~1.5x scale

wind_availability_gen1 = np.clip(wind_availability_gen1 / max_possible_speed, 0.4, 1.0) # Ensure at least 40% availability

wind_availability_gen2 = np.clip(wind_availability_gen2 / max_possible_speed, 0.4, 1.0) # Ensure at least 40% availability

# Create the availability matrix

availability_matrix = np.ones((len(gen_IDs), hours))

availability_matrix[0, :] = wind_availability_gen1 # Generator 1 (wind)

availability_matrix[1, :] = solar_availability # Generator 2 (solar)

 

# DETERMINE RAMP RATES FOR GENERATORS 3, 4 AND 5

ramp_rates = {}

for gen_id, tau, delay, P_initial, P_target in zip(

    [3, 4, 5], [1.6, 1.7, 2.5], [0.55, 0.6, 1.0], [0, 0, 0], [gen_capacities[2], gen_capacities[3], gen_capacities[4]]

):

    ramp_rates[gen_id] = simulate_dde(tau, delay, P_initial, P_target)

print("Ramp rates (MW/hour):", ramp_rates)

 

 

line_ratings_vector = line_ratings.values.flatten()

dispatch_cost, q_supply_table, PF_values = run_optimization(line_ratings_vector)

# dispatch_cost, q_supply_table, PF_values = resolve_bottlenecks(line_ratings_vector)

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

 

# Below code for getting surplus plot

total_demand = node_demands.sum(axis=1)

total_supply = q_supply_table.sum(axis=0)

# Compute surplus energy as gen1_cap minus the first row of q_supply_table

surplus_energy_gen1 = np.maximum(q_supply_table.iloc[0, :], (gen_capacities[0] * wind_availability_gen1) - q_supply_table.iloc[0, :])

surplus_energy_gen2 = np.maximum(q_supply_table.iloc[1, :], (gen_capacities[1] * solar_availability) - q_supply_table.iloc[1, :])

tot_potential_renewable_gen = surplus_energy_gen1 + surplus_energy_gen2

 

tot_non_renewable_generation = q_supply_table.iloc[2:5, :].sum(axis=0)  # 1x168 row vector

tot_potential_gen = tot_potential_renewable_gen + tot_non_renewable_generation

 

x_values = np.arange(hours)

# PLOT RESULTS

plt.figure(figsize=(12, 6))

plt.plot(total_demand, label="Total Demand (MW)", color="red", linewidth=2)

plt.plot(x_values, tot_potential_gen, label="Total Potential Generation(MW)", color="green", linewidth=2)

plt.fill_between(range(hours), tot_potential_gen, total_demand, where=(tot_potential_gen >= total_demand),

                 color="green", alpha=0.3, label="Surplus Renewable Energy")

plt.xlabel("Hour", fontsize=28)

plt.ylabel("Power (MW)", fontsize=28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.title("Wind Generation vs Demand w/ Transmission Constraints", fontsize=32, fontweight='bold')

plt.legend(fontsize=24)

plt.grid(True)

plt.tight_layout()

plt.show()




## NET SAVINGS VS TURBINE COSTS--------------------------------------------------------------------------------------------------------------

 

# Compute the correct renewable surplus energy

final_surplus_energy_gen1 = (gen_capacities[0] * wind_availability_gen1) - q_supply_table.iloc[0, :]

final_surplus_energy_gen2 = (gen_capacities[1] * solar_availability) - q_supply_table.iloc[1, :]

final_total_surplus = final_surplus_energy_gen1 + final_surplus_energy_gen2  # Only unused renewable energy

 

# Expansion levels for wind turbines (starting at 2x)

expansion_levels = range(2, 6)  # Doubling, tripling, etc., up to 5x initial capacity

construction_cost_per_mw = 1_000_000  # £1M per MW

# Marginal cost of coal/gas generation (approximation)

fossil_fuel_marginal_cost = 216  # £/MWh

 

# Initial generator capacities

solar_capacity = 350  # MW

initial_wind_capacity = 350  # MW

gen_capacities = [initial_wind_capacity, solar_capacity]

 

# Battery storage cost (mid-range estimate)

battery_cost_per_mwh = 133  # £/kWh (or £300,000 per MWh for battery capacity)

savings = []

construction_costs = []

battery_costs = []

 

for multiplier in expansion_levels:

 

    # Updated wind capacity (doubling, tripling, etc.)

    wind_capacity = initial_wind_capacity * multiplier

    gen_capacities[0] = wind_capacity  # Update wind capacity in generator capacities

   

    # Renewable generation

    wind_generation = wind_capacity * wind_availability_gen1

    solar_generation = solar_capacity * solar_availability  # Solar remains fixed

    total_renewable_generation = wind_generation + solar_generation

   

    # Surplus energy

    surplus_energy = np.maximum(total_renewable_generation - total_demand, 0)  # Surplus in MW

    total_surplus_energy_mwh = np.sum(surplus_energy)  # Convert to MWh for the week

   

    # Debug surplus calculation

    print(f"Multiplier: {multiplier}")

    print(f"Total Renewable Generation: {np.sum(total_renewable_generation):.2f} MW")

    print(f"Total Demand: {np.sum(total_demand):.2f} MW")

    print(f"Surplus Energy (MWh): {total_surplus_energy_mwh:.2f}")

   

    # Savings from surplus energy

    surplus_savings = total_surplus_energy_mwh * fossil_fuel_marginal_cost

   

    # Cost of storing surplus energy in a battery

    storage_cost = total_surplus_energy_mwh * battery_cost_per_mwh

   

    # Adjust savings by subtracting storage cost

    net_savings = surplus_savings - storage_cost

    savings.append(net_savings)

   

    # Battery cost tracking

    battery_costs.append(storage_cost)

   

    # Construction cost for wind farm expansion (incremental cost)

    additional_wind_capacity = (multiplier - 1) * initial_wind_capacity  # Additional MW added

    wind_construction_cost = additional_wind_capacity * construction_cost_per_mw

    construction_costs.append(wind_construction_cost)

 

# Plot Savings vs. Costs

plt.figure(figsize=(12, 6))

plt.plot(expansion_levels, savings, label="Net Savings (£)", marker="o", color="green")

plt.plot(expansion_levels, construction_costs, label="Wind Turbine Construction Cost (£)", marker="o", color="red")

plt.xlabel("Wind Turbine Expansion Factor", fontsize=28)

plt.ylabel("Cost (£)", fontsize=28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.title("Feasibility of Wind Turbine Expansion w/ Storage Costs", fontsize = 32, fontweight = 'bold')
ax = plt.gca()
plt.ticklabel_format(style='scientific', axis='y', scilimits=(6, 9))
ax.yaxis.offsetText.set_fontsize(28)  # Make the offset (1e6) larger
plt.legend(fontsize=24)

plt.grid(True)

plt.tight_layout()

plt.show()

 

# Plot Savings Only

plt.figure(figsize=(12, 6))

plt.plot(expansion_levels, savings, label="Net Savings (£)", marker="o", color="green")

plt.xlabel("Wind Turbine Expansion Factor", fontsize=28)

plt.ylabel("Net Savings (£)", fontsize=28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.title("Net Savings from Wind Farm Expansion w/ Storage Costs", fontsize=32, fontweight= 'bold')
ax = plt.gca()
plt.ticklabel_format(style='scientific', axis='y', scilimits=(6, 6))
ax.yaxis.offsetText.set_fontsize(28)  # Make the offset (1e6) larger
plt.grid(True)

plt.tight_layout()

plt.show()

 

#--------------------------------------------------------------------------------------------------------

# Provided data (adjusted to use actual savings)

wind_multipliers = [2, 3, 4, 5]

construction_costs = [175_000_000, 700_000_000, 1_050_000_000, 1_400_000_000]  # £

# Replace these values with your calculated savings

weekly_savings = savings  # Use actual savings from the simulation

 

# Exclude the first multiplier from both lists

# wind_multipliers_excluded = wind_multipliers[0:]

# construction_costs_excluded = construction_costs[0:]

# weekly_savings_excluded = weekly_savings[0:]

 

# Calculate breakeven weeks excluding the first value

breakeven_weeks_excluded = []

for i in range(len(wind_multipliers)):

    if weekly_savings[i] > 0:

        breakeven_weeks_excluded.append(construction_costs[i] / weekly_savings[i])

    else:

        breakeven_weeks_excluded.append(float('inf'))  # Handle division by zero case

 

# Debug breakeven weeks

print("Breakeven weeks calculated (excluding first value):", breakeven_weeks_excluded)

 

# Plotting Breakeven Weeks

plt.figure(figsize=(12, 6))

plt.plot(wind_multipliers, breakeven_weeks_excluded, marker='o', linestyle='-', color='purple', label="Breakeven Weeks")

plt.axhline(y=52, color='blue', linestyle='--', label="1 Year (52 Weeks)")

plt.axhline(y=260, color='green', linestyle='--', label="5 Years (260 Weeks)")

plt.axhline(y=520, color='orange', linestyle='--', label="10 Years (520 Weeks)")

 

# Labels and legend

plt.xlabel("Wind Farm Multiplier (x Initial Capacity)", fontsize=28)

plt.ylabel("Weeks to Breakeven", fontsize=28)

plt.title("Weeks to Break-even for Wind Farm Expansion", fontsize=32, fontweight= 'bold')
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.legend(fontsize=24)
# (Excluding Initial Multiplier)
plt.grid(True)

plt.tight_layout()

plt.show()