import numpy as np
import pandas as pd
import cvxpy as cp
from ddeint import ddeint
import matplotlib.pyplot as plt

# INPUT DATA
generator_data = pd.read_csv("provided_material/generators.csv", index_col=0)
node_demands = pd.read_csv("provided_material/hourlydemandbynode.csv", index_col=0)
shift_factor_matrix = pd.read_csv("provided_material/shiftfactormatrix.csv", index_col=0)
line_ratings = pd.read_csv("provided_material/lineratings.csv", index_col=0)

# Extract generator and demand data
gen_IDs = np.array(generator_data.loc[:, "NODE"].astype(int))
gen_marginal_costs = np.array(generator_data.loc[:, "MC"])  # (£/MWh)
gen_capacities = np.array(generator_data.loc[:, "CAP"])  # (MW)

demand_IDs = np.array(node_demands.columns.astype(int))
node_IDs = np.array(shift_factor_matrix.columns.astype(int))

# Generator and demand identity matrices
gen_indices = {g_ID: np.where(node_IDs == g_ID)[0][0] for g_ID in gen_IDs}
gen_identity_mat = np.zeros((len(node_IDs), len(gen_IDs)))
gen_identity_mat[list(gen_indices.values()), np.arange(len(gen_IDs))] = 1

demand_indices = {d_ID: np.where(node_IDs == d_ID)[0][0] for d_ID in demand_IDs}
demand_identity_mat = np.zeros((len(node_IDs), len(demand_IDs)))
demand_identity_mat[list(demand_indices.values()), np.arange(len(demand_IDs))] = 1

# Simulate DDE to calculate ramp rates for generators 3, 4, and 5
def simulate_dde(tau, delay, P_initial, P_target, duration=10, steps=1000):
    time = np.linspace(0, duration, steps)

    def model(P, t):
        return (P_target - P(t - delay)) / tau

    def history(t):
        return P_initial

    response = ddeint(model, history, time)
    ramp_rate = (response[-1] - response[0]) / duration
    return abs(ramp_rate)

# Define ramp rates for generators 3, 4, and 5
ramp_rates = {}
for gen_id, tau, delay, P_initial, P_target in zip(
    [3, 4, 5], [1.5, 2.0, 2.5], [0.5, 0.7, 1.0], [0, 0, 0], [gen_capacities[2], gen_capacities[3], gen_capacities[4]]
):
    ramp_rates[gen_id] = simulate_dde(tau, delay, P_initial, P_target)

print("Ramp rates (MW/hour):", ramp_rates)

# Helper function to set availability constraints
def set_availability(scenario):
    """
    Set availability constraints for the generators based on the scenario.
    """
    hours = node_demands.shape[0]
    availability_matrix = np.ones((len(gen_IDs), hours))

    if scenario == "no_constraints":
        pass  # All generators available without constraints

    elif scenario == "one_wind_one_solar":
        availability_matrix[0, :] = np.clip(np.random.weibull(2.0, hours) * 0.8, 0.4, 1.0)  # Wind (GEN 1)
        availability_matrix[1, :] = 0.5 * (1 + np.sin(2 * np.pi * np.arange(hours) / 24 - np.pi / 2))  # Solar (GEN 2)

    elif scenario == "two_wind":
        availability_matrix[0, :] = np.clip(np.random.weibull(2.0, hours) * 0.8, 0.4, 1.0)  # Wind (GEN 1)
        availability_matrix[1, :] = np.clip(np.random.weibull(2.0, hours) * 0.8, 0.4, 1.0)  # Wind (GEN 2)

    elif scenario == "two_solar":
        availability_matrix[0, :] = 0.5 * (1 + np.sin(2 * np.pi * np.arange(hours) / 24 - np.pi / 2))  # Solar (GEN 1)
        availability_matrix[1, :] = 0.5 * (1 + np.sin(2 * np.pi * np.arange(hours) / 24 - np.pi / 2))  # Solar (GEN 2)

    elif scenario == "one_solar_two_wind":
        availability_matrix[0, :] = np.clip(np.random.weibull(2.0, hours) * 0.8, 0.4, 1.0)  # Wind (GEN 1)
        availability_matrix[1, :] = 0.5 * (1 + np.sin(2 * np.pi * np.arange(hours) / 24 - np.pi / 2))  # Solar (GEN 2)
        availability_matrix[2, :] = np.clip(np.random.weibull(2.0, hours) * 0.8, 0.4, 1.0)  # Wind (GEN 3)

    return availability_matrix

# Function to solve the optimization problem for a given scenario
def solve_scenario(scenario):
    # Set generator data
    availability_matrix = set_availability(scenario)
    q_supply = cp.Variable((len(gen_IDs), node_demands.shape[0]), nonneg=True)

    # Constraints
    constraints = []
    constraints.append(cp.sum(q_supply, axis=0) == cp.sum(node_demands.values.T, axis=0))  # Demand balance

    # Line flow constraints
    net_injections = gen_identity_mat @ q_supply - demand_identity_mat @ node_demands.values.T
    PF = shift_factor_matrix.values @ net_injections
    finite_mask = np.isfinite(line_ratings.values.flatten())
    constraints += [
        PF[finite_mask] <= line_ratings.values.flatten()[finite_mask][:, None],
        PF[finite_mask] >= -line_ratings.values.flatten()[finite_mask][:, None],
    ]

    # Generator capacity constraints
    constraints += [q_supply <= gen_capacities[:, None] * availability_matrix]

    # Ramping constraints for generators 3, 4, and 5
    for t in range(1, node_demands.shape[0]):
        for gen_id, ramp_rate in ramp_rates.items():
            gen_idx = gen_id - 1  # Adjust for zero-based indexing
            constraints.append(q_supply[gen_idx, t] - q_supply[gen_idx, t - 1] <= ramp_rate)
            constraints.append(q_supply[gen_idx, t - 1] - q_supply[gen_idx, t] <= ramp_rate)

    # Objective function
    objective = cp.Minimize(cp.sum(cp.multiply(gen_marginal_costs[:, None], q_supply)))

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CBC)

    if problem.status == cp.OPTIMAL:
        return problem.value, np.allclose(cp.sum(q_supply, axis=0).value, cp.sum(node_demands.values.T, axis=0).value)
    else:
        return None, False  # If not optimal

# Scenarios
scenarios = [
    "no_constraints",
    "one_wind_one_solar",
    "two_wind",
    "two_solar",
    "one_solar_two_wind",
]

# Solve for all scenarios
results = {}
for scenario in scenarios:
    print(f"Solving for scenario: {scenario}")
    cost, demand_met = solve_scenario(scenario)
    if cost is not None:
        results[scenario] = {"Cost": cost, "Demand Met": demand_met}
    else:
        results[scenario] = {"Cost": float("inf"), "Demand Met": False}

# Plot results
costs = [results[scenario]["Cost"] for scenario in scenarios]
labels = ["No Constraints", "1 Wind, 1 Solar", "2 Wind", "2 Solar", "1 Solar, 2 Wind"]

plt.figure(figsize=(10, 6))
plt.bar(labels, costs, color=["blue", "green", "cyan", "orange", "purple"])
plt.title("Total Dispatch Costs Across Scenarios")
plt.ylabel("Total Dispatch Cost (£)")
plt.xlabel("Scenarios")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# Print results
for scenario, result in results.items():
    print(f"Scenario: {scenario}, Cost: £{result['Cost']:.2f}, Demand Met: {result['Demand Met']}")
