import pandas as pd
import numpy as np
import cvxpy as cp
from ddeint import ddeint
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------------------------------------------
# INPUT DATA
generator_data = pd.read_csv("generators.csv", index_col=0)
node_demands = pd.read_csv("hourlydemandbynode.csv", index_col=0)
shift_factor_matrix = pd.read_csv("shiftfactormatrix.csv", index_col=0)
line_ratings = pd.read_csv("lineratings.csv", index_col=0)
network_edges = pd.read_csv("network_edges.csv", index_col=0)

# -------------------------------------------------------------------------------------------------------
# EXTRACT REQUIRED INFO FROM INPUT DATA
gen_IDs = np.array(generator_data.loc[:, "NODE"].astype(int))
gen_marginal_costs = np.array(generator_data.loc[:, "MC"])  # (£/MWh)
gen_capacities = np.array(generator_data.loc[:, "CAP"])  # (MW)

demand_IDs = np.array(node_demands.columns.astype(int))
node_IDs = np.array(shift_factor_matrix.columns.astype(int))

# -------------------------------------------------------------------------------------------------------
# FUNCTION TO SIMULATE DDE AND DETERMINE RAMP RATES
def simulate_dde(tau, delay, P_initial, P_target, duration=10, steps=1000):
    time = np.linspace(0, duration, steps)

    def model(P, t):
        return (P_target - P(t - delay)) / tau

    def history(t):
        return P_initial

    response = ddeint(model, history, time)
    ramp_rate = (response[-1] - response[0]) / duration
    return abs(ramp_rate)

# Calculate ramp rates
ramp_rates = {}
for gen_id, tau, delay, P_initial, P_target in zip(
    [3, 4, 5], [1.5, 2.0, 2.5], [0.5, 0.7, 1.0], [0, 0, 0], [gen_capacities[2], gen_capacities[3], gen_capacities[4]]
):
    ramp_rates[gen_id] = simulate_dde(tau, delay, P_initial, P_target)
print("Ramp rates (MW/hour):", ramp_rates)

# -------------------------------------------------------------------------------------------------------
# GENERATOR AND DEMAND MATRICES
gen_indices = {g_ID: np.where(node_IDs == g_ID)[0][0] for g_ID in gen_IDs}
gen_identity_mat = np.zeros((len(node_IDs), len(gen_IDs)))
gen_identity_mat[list(gen_indices.values()), np.arange(len(gen_IDs))] = 1

demand_indices = {d_ID: np.where(node_IDs == d_ID)[0][0] for d_ID in demand_IDs}
demand_identity_mat = np.zeros((len(node_IDs), len(demand_IDs)))
demand_identity_mat[list(demand_indices.values()), np.arange(len(demand_IDs))] = 1

# -------------------------------------------------------------------------------------------------------
# HOURLY AVAILABILITY FOR GENERATORS 1 AND 2
hours = node_demands.shape[0]
time = np.arange(hours)
# Solar availability as a sine wave (Generator 2)
solar_availability = 0.5 * (1 + np.sin(2 * np.pi * time / 24 - np.pi / 2))  # Sine wave shifted to make nighttime 0
solar_availability = np.clip(solar_availability, 0, 1)  # Ensure values are between 0 and 1

# Generate Weibull-distributed wind availability
np.random.seed(42)  
shape, scale = 2.3, 9.5  # Example Weibull parameters
wind_availability_gen1 = np.random.weibull(shape, hours) * scale
wind_availability_gen2 = np.random.weibull(shape, hours) * scale

max_possible_speed = scale * 1.5  # Assume maximum wind speed reaches ~1.5x scale
wind_availability_gen1 = np.clip(wind_availability_gen1 / max_possible_speed, 0.4, 1.0) # Ensure at least 40% availability
wind_availability_gen2 = np.clip(wind_availability_gen2 / max_possible_speed, 0.4, 1.0) # Ensure at least 40% availability

# Create the availability matrix
availability_matrix = np.ones((len(gen_IDs), hours))
availability_matrix[0, :] = wind_availability_gen1 # Generator 1 (solar)
availability_matrix[1, :] = wind_availability_gen2  # Generator 2 (wind)


# -------------------------------------------------------------------------------------------------------
# DECISION VARIABLES
q_supply = cp.Variable((len(gen_IDs), hours), nonneg=True)

# -------------------------------------------------------------------------------------------------------
# CONSTRAINTS
constraints = []

# Total generation must meet demand
constraints.append(cp.sum(q_supply, axis=0) == cp.sum(node_demands.values.T, axis=0))

# Line flow constraints
net_injections = gen_identity_mat @ q_supply - demand_identity_mat @ node_demands.values.T
PF = shift_factor_matrix.values @ net_injections

# Reshape R_PU for broadcasting with PF
R_PU_reshaped = network_edges['R_PU'].values.reshape(-1, 1)  # Reshape R_PU to (532, 1) 

# Calculate transmission losses (elementwise multiplication of squared power flow and resistance)
line_losses = cp.sum(cp.multiply(cp.square(PF), R_PU_reshaped))

# Ensure power flow does not exceed line ratings
finite_mask = np.isfinite(line_ratings.values.flatten())
constraints += [
    PF[finite_mask] <= line_ratings.values.flatten()[finite_mask][:, None],
    PF[finite_mask] >= -line_ratings.values.flatten()[finite_mask][:, None],
]

# Generator capacity constraints
constraints += [q_supply <= gen_capacities[:, None] * availability_matrix]

# Remaining demand constraints
remaining_demand = cp.sum(node_demands.values.T, axis=0) - cp.sum(q_supply[:2, :], axis=0)
constraints.append(cp.sum(q_supply[2:, :], axis=0) >= remaining_demand)

# Capacity constraints for Generators 3, 4, and 5
constraints += [q_supply[2:, :] <= gen_capacities[2:, None]]

# Ramping constraints
for t in range(1, hours):
    for g, ramp_rate in ramp_rates.items():
        gen_idx = g - 1
        constraints.append(q_supply[gen_idx, t] - q_supply[gen_idx, t - 1] <= ramp_rate)
        constraints.append(q_supply[gen_idx, t - 1] - q_supply[gen_idx, t] <= ramp_rate)

# -------------------------------------------------------------------------------------------------------
# OBJECTIVE FUNCTION
# Minimize generation costs and transmission losses
objective = cp.Minimize(cp.sum(cp.multiply(gen_marginal_costs[:, None], q_supply)) + line_losses)

# -------------------------------------------------------------------------------------------------------
# SOLVE THE OPTIMIZATION PROBLEM
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-3, eps_rel=1e-3, verbose=True, max_iter=20000)


# -------------------------------------------------------------------------------------------------------
# RESULTS
if problem.status == cp.OPTIMAL:
    print("OPTIMAL SOLUTION FOUND!")
    print(f"Dispatch costs (£): {problem.value:.2f}")
    
    # Convert optimal generator dispatch to a DataFrame
    q_supply_table = pd.DataFrame(q_supply.value, index=gen_IDs, columns=node_demands.index)

    # Print results
    print("Optimal generator dispatch (MW):")
    print(q_supply_table)

    # -------------------------------------------------------------------------------------------------------
    # PLOT GENERATOR DISPATCH
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

    # -------------------------------------------------------------------------------------------------------
    # CALCULATE LOCATIONAL MARGINAL PRICING (LMP)
    # debugging doesn't work for LMP calcultion as of now. prints dispatch costs with losses though
    print("Generator Node IDs:", gen_IDs)
    print("Mapped Generator Column Indices:", gen_column_indices)
    print("Max column index:", max(gen_column_indices))  # Should be < 168

    # Extract only the relevant columns from the shift factor matrix
    try:
        gen_column_indices = [np.where(node_IDs == gen_id)[0][0] for gen_id in gen_IDs]
        relevant_shift_factors = shift_factor_matrix.iloc[:, gen_column_indices]  # Expected shape: (532, num_generators)
    except Exception as e:
        print("Error extracting relevant shift factors:", e)
        exit()

    print("Shape of relevant_shift_factors:", relevant_shift_factors.shape) 

    # Energy price (5 generators × 168 hours)
    energy_price = np.repeat(gen_marginal_costs[:, None], hours, axis=1)
    print("Shape of energy_price:", energy_price.shape)  # Expected: (5, 168)

    # Congestion price calculation
    try:
        congestion_price = np.dot(relevant_shift_factors.values, q_supply.value)  # Expected: (532, 168)
        print("Shape of congestion_price:", congestion_price.shape)  
    except Exception as e:
        print("Error computing congestion_price:", e)
        exit()

    # Compute congestion price at nodes
    try:
        congestion_price_at_nodes = relevant_shift_factors.T @ congestion_price  # Expected: (428, 168)
        congestion_price_at_nodes = congestion_price_at_nodes.to_numpy()
        print("Shape of congestion_price_at_nodes:", congestion_price_at_nodes.shape)
    except Exception as e:
        print("Error computing congestion_price_at_nodes:", e)
        exit()

    # Extract congestion price for generator nodes
    try:
        congestion_price_at_gens = congestion_price_at_nodes[:, gen_column_indices]  # Expected: (5, 168)
        print("Shape of congestion_price_at_gens:", congestion_price_at_gens.shape)
    except Exception as e:
        print("Error extracting congestion_price_at_gens:", e)
        exit()

    # Ensure losses price is broadcastable
    try:
        losses_price = np.full((len(gen_IDs), hours), line_losses.value)  # Expected: (5, 168)
        print("Shape of losses_price:", losses_price.shape)
    except Exception as e:
        print("Error computing losses_price:", e)
        exit()

    # Compute final LMP
    try:
        LMP = energy_price + congestion_price_at_gens + losses_price  # Expected: (5, 168)
        print("Shape of LMP:", LMP.shape)
    except Exception as e:
        print("Error computing LMP:", e)
        exit()

    # -------------------------------------------------------------------------------------------------------
    # PLOT LOCATIONAL MARGINAL PRICING (LMP)
    plt.figure(figsize=(12, 6))
    sns.heatmap(LMP, cmap="YlGnBu", annot=False, fmt=".2f", xticklabels=node_demands.index, yticklabels=gen_IDs)
    plt.title("Locational Marginal Pricing (LMP) for Each Generator at Each Node")
    plt.xlabel("Hour")
    plt.ylabel("Generator")
    plt.colorbar(label="LMP (£/MWh)")
    plt.tight_layout()
    plt.show()

else:
    print("NO OPTIMAL SOLUTION FOUND!")
