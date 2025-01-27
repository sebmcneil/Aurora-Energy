import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

# INPUT DATA
generator_data = pd.read_csv("provided_material/generators.csv", index_col=0)
node_demands = pd.read_csv("provided_material/hourlydemandbynode.csv", index_col=0)  # 168 hours by 20 demand nodes
shift_factor_matrix = pd.read_csv("provided_material/shiftfactormatrix.csv", index_col=0)  # Have LINE_ID as row index of the dataframe
line_ratings = pd.read_csv("provided_material/lineratings.csv", index_col=0)  # 532x1 vector

# Extract necessary data
gen_IDs = np.array(generator_data.loc[:, "NODE"].astype(int))
gen_marginal_costs = np.array(generator_data.loc[:, "MC"])
gen_capacities = np.array(generator_data.loc[:, "CAP"])
demand_IDs = np.array(node_demands.columns.astype(int))
node_IDs = np.array(shift_factor_matrix.columns.astype(int))

gen_indices = {g_ID: np.where(node_IDs == g_ID)[0][0] for g_ID in gen_IDs}
gen_identity_mat = np.zeros((len(node_IDs), len(gen_IDs)))
gen_identity_mat[list(gen_indices.values()), np.arange(len(gen_IDs))] = 1

demand_indices = {d_ID: np.where(node_IDs == d_ID)[0][0] for d_ID in demand_IDs}
demand_identity_mat = np.zeros((len(node_IDs), len(demand_IDs)))
demand_identity_mat[list(demand_indices.values()), np.arange(len(demand_IDs))] = 1

original_line_ratings = line_ratings.values.flatten()
finite_mask = np.isfinite(original_line_ratings)

# Optimization function to compute dispatch cost and PF_values
def solve_optimization(line_ratings_vector):
    """
    Solve the optimization problem and return dispatch cost and power flows.

    Args:
        line_ratings_vector (ndarray): Line ratings for all lines (num_lines).

    Returns:
        tuple: (dispatch_cost, PF_values)
    """
    q_supply = cp.Variable((len(gen_IDs), len(node_demands)), nonneg=True)
    objective = cp.Minimize(cp.sum(cp.multiply(gen_marginal_costs[:, None], q_supply)))

    constraints = []
    constraints.append(cp.sum(q_supply, axis=0) == cp.sum(node_demands.values.T, axis=0))  # Demand balance
    net_injections = gen_identity_mat @ q_supply - demand_identity_mat @ node_demands.values.T
    PF = shift_factor_matrix.values @ net_injections  # Power flows

    finite_mask = np.isfinite(line_ratings_vector)
    constraints += [
        PF[finite_mask] <= line_ratings_vector[finite_mask][:, None],
        PF[finite_mask] >= -line_ratings_vector[finite_mask][:, None]
    ]
    constraints += [q_supply <= gen_capacities[:, None]]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CBC)

    if problem.status == cp.OPTIMAL:
        return problem.value, PF.value
    else:
        raise RuntimeError("Optimization failed!")

# Marginal congestion analysis
def marginal_congestion_analysis(PF_values, line_ratings_vector):
    """
    Calculate a congestion metric for each line based on its power flow relative to its rating.

    Args:
        PF_values (ndarray): Power flow values for all lines across all hours (num_lines x num_hours).
        line_ratings_vector (ndarray): Line ratings for all lines (num_lines).

    Returns:
        pd.DataFrame: Congestion metrics for each line.
    """
    finite_mask = np.isfinite(line_ratings_vector)  # Exclude infinite-rated lines
    congestion_metric = np.zeros_like(line_ratings_vector)

    for line in range(PF_values.shape[0]):
        if finite_mask[line]:  # Only consider finite-rated lines
            congestion_metric[line] = np.sum(
                np.abs(PF_values[line, :]) / line_ratings_vector[line]
            )

    # Create DataFrame for results
    congestion_results = pd.DataFrame({
        "Congestion Metric": congestion_metric,
    }, index=shift_factor_matrix.index)  # Use line IDs as index

    # Sort lines by congestion metric in descending order
    congestion_results = congestion_results.sort_values(by="Congestion Metric", ascending=False)

    return congestion_results

# Function to compute total dispatch cost and dynamic upgrades
def compute_total_cost_at_upgrades(solve_optimization_func, original_line_ratings, increment_steps, top_n=5):
    """
    Compute the total dispatch cost at each upgrade step and perform dynamic upgrades.

    Args:
        solve_optimization_func (function): Function to solve the optimization problem.
        original_line_ratings (ndarray): Original line ratings (num_lines).
        increment_steps (list): List of percentages to increment line ratings.
        top_n (int): Number of top congested lines to upgrade in each step.

    Returns:
        list: Total dispatch costs at each upgrade step.
    """
    current_ratings = original_line_ratings.copy()
    total_costs = []

    for step, increment in enumerate(increment_steps, start=1):

        # Solve optimization with current ratings
        dispatch_cost, PF_values = solve_optimization_func(current_ratings)

        # Save total dispatch cost
        total_costs.append(dispatch_cost)

        # Get top N most congested lines and upgrade their ratings
        congestion_results = marginal_congestion_analysis(PF_values, current_ratings)
        top_congested_lines = congestion_results.head(top_n).index

        for line_id in top_congested_lines:
            line_index = shift_factor_matrix.index.get_loc(line_id)
            current_ratings[line_index] *= (1 + increment)

    return total_costs

# Define upgrade increments (10% per step for 5 steps)
increment_steps = [0.1] * 5  # Each step increases ratings by 10%

# Compute total dispatch costs at each step
total_costs = compute_total_cost_at_upgrades(
    solve_optimization,
    original_line_ratings,
    increment_steps,
    top_n=5
)

# # Plot the total dispatch cost at each step
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(total_costs) + 1), total_costs, marker='o')
# plt.title("Total Dispatch Cost Across Upgrade Steps")
# plt.xlabel("Upgrade Step")
# plt.ylabel("Total Dispatch Cost (£)")
# plt.xticks(range(1, len(total_costs) + 1), labels=[f"Step {step}" for step in range(1, len(total_costs) + 1)])
# plt.grid()
# plt.show()

# # Adjust the figure to make all the text larger
# plt.figure(figsize=(12, 8))
# plt.plot(range(1, len(total_costs) + 1), total_costs, marker='o')
# plt.title("Total Dispatch Cost Across Upgrade Steps", fontsize=24,)
# plt.xlabel("Upgrade Step", fontsize=20, labelpad=10)
# plt.ylabel("Total Dispatch Cost (£)", fontsize=20, labelpad=10)
# plt.xticks(range(1, len(total_costs) + 1), labels=[f"Step {step}" for step in range(1, len(total_costs) + 1)], fontsize=14)
# plt.yticks(fontsize=18)
# plt.grid()
# plt.show()

# # Plot with larger fonts and markers
# plt.figure(figsize=(12, 8))
# plt.plot(
#     range(1, len(total_costs) + 1), 
#     total_costs, 
#     marker='o', 
#     markersize=10, 
#     linewidth=2, 
#     label="Dispatch Cost"
# )
# plt.title("Total Dispatch Cost Across Upgrade Steps", fontsize=22, fontweight='bold')
# plt.xlabel("Upgrade Step", fontsize=18, labelpad=15)
# plt.ylabel("Total Dispatch Cost (£)", fontsize=18, labelpad=15)
# plt.xticks(range(1, len(total_costs) + 1), labels=[f"Step {step}" for step in range(1, len(total_costs) + 1)], fontsize=16)
# plt.yticks(fontsize=16)
# plt.grid(True, linestyle='--', linewidth=0.5)
# plt.legend(fontsize=16)
# plt.show()

# Plot with adjustments for scientific notation and removing the legend
plt.figure(figsize=(12, 8))
plt.plot(
    range(1, len(total_costs) + 1), 
    total_costs, 
    marker='o', 
    markersize=10, 
    linewidth=2, 
    label="Dispatch Cost"
)
plt.title("Total Dispatch Cost Across Upgrade Steps", fontsize=22, fontweight='bold')
plt.xlabel("Upgrade Step", fontsize=18, labelpad=15)
plt.ylabel("Total Dispatch Cost (£)", fontsize=18, labelpad=15)
plt.xticks(range(1, len(total_costs) + 1), labels=[f"Step {step}" for step in range(1, len(total_costs) + 1)], fontsize=16)
plt.yticks(fontsize=16)

# Adjust scientific notation and increase the font size for 1e6
ax = plt.gca()
plt.ticklabel_format(style='scientific', axis='y', scilimits=(6, 6))
ax.yaxis.offsetText.set_fontsize(16)  # Make the offset (1e6) larger

# Remove the legend
plt.legend().remove()

plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()


# Print total costs
print("\nTotal Dispatch Costs at Each Upgrade Step:")
for step, cost in enumerate(total_costs, start=1):
    print(f"Step {step}: £{cost:.2f}")


