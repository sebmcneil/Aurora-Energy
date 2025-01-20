import numpy as np
import pandas as pd
import cvxpy as cp

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
        return problem.value, PF.value  # Return dispatch cost and power flows
    else:
        raise RuntimeError("Optimization failed!")



def incremental_dispatch_cost_sensitivity(line_ratings_vector, shift_factor_matrix, run_optimization_func, increment=0.05):
    """
    Evaluate the sensitivity of dispatch costs to line capacity increases.

    Args:
        line_ratings_vector (ndarray): Line ratings for all lines (num_lines).
        shift_factor_matrix (pd.DataFrame): Shift factor matrix.
        run_optimization_func (function): Function to solve the optimization problem.
        increment (float): Percentage increase for each line rating.

    Returns:
        pd.DataFrame: Dispatch cost reduction for each line.
    """
    # Get the base cost
    base_cost, _ = run_optimization_func(line_ratings_vector)
    sensitivity_results = []

    for line_id in shift_factor_matrix.index:
        # Create a copy of the line ratings
        updated_ratings = line_ratings_vector.copy()
        
        # Increase the capacity of the current line
        idx = np.where(shift_factor_matrix.index == line_id)[0][0]
        updated_ratings[idx] *= (1 + increment)
        
        # Solve the optimization problem with the updated ratings
        new_cost, _ = run_optimization_func(updated_ratings)
        cost_reduction = base_cost - new_cost
        
        sensitivity_results.append({"Line ID": line_id, "Cost Reduction": cost_reduction})

    # Create a DataFrame for results
    sensitivity_df = pd.DataFrame(sensitivity_results)
    sensitivity_df = sensitivity_df.sort_values(by="Cost Reduction", ascending=False)

    print("\nIncremental Dispatch Cost Sensitivity:")
    print(sensitivity_df.head(10))  # Show top 10 lines with the largest cost reductions
    return sensitivity_df




# Solve optimization to compute PF_values (needed for other analyses)
_, PF_values = solve_optimization(original_line_ratings)

# Perform incremental dispatch cost sensitivity analysis
sensitivity_results = incremental_dispatch_cost_sensitivity(
    line_ratings.values.flatten(),
    shift_factor_matrix,
    solve_optimization,
    increment=0.05
)




