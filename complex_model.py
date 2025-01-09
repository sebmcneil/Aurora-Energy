import pandas as pd
import numpy as np
import cvxpy as cp

# Load data
generators = pd.read_csv("generators.csv", index_col=0)  # Generator data
node_demands = pd.read_csv("hourlydemandbynode.csv", index_col=0)  # Hourly demand
shift_factor_matrix = pd.read_csv("shiftfactormatrix.csv", index_col=0)  # PTDF
line_ratings = pd.read_csv("lineratings.csv", index_col=0)  # Line ratings

# Extract data
gen_ids = generators["NODE"].astype(int).values
gen_costs = generators["MC"].values  # (£/MWh)
gen_caps = generators["CAP"].values  # (MW)
demand_ids = node_demands.columns.astype(int).values
node_ids = shift_factor_matrix.columns.astype(int).values
line_limits = line_ratings["RATING_MW"].replace("inf", np.inf).values

# Decision variables
q_supply = cp.Variable((len(gen_ids), len(node_demands)), nonneg=True)  # Power from generators (MW)

# Objective: Minimize total cost
objective = cp.Minimize(cp.sum(cp.multiply(gen_costs[:, None], q_supply)))

# Constraints
constraints = []

# Power balance
for t in range(len(node_demands)):
    # Create net injection as a CVXPY expression
    net_injections = cp.Constant(np.zeros(len(node_ids)))
    for i, gen_id in enumerate(gen_ids):
        gen_idx = np.where(node_ids == gen_id)[0][0]
        net_injections += cp.Constant(np.eye(len(node_ids))[:, gen_idx]) * q_supply[i, t]

    for i, demand_id in enumerate(demand_ids):
        demand_idx = np.where(node_ids == demand_id)[0][0]
        net_injections -= cp.Constant(np.eye(len(node_ids))[:, demand_idx]) * node_demands.iloc[t, i]

    # Compute power flows
    power_flows = shift_factor_matrix.values @ net_injections
    for line in range(len(line_limits)):
        if line_limits[line] != np.inf:
            constraints.append(power_flows[line] <= line_limits[line])
            constraints.append(power_flows[line] >= -line_limits[line])

# Generator constraints
for i, cap in enumerate(gen_caps):
    for t in range(len(node_demands)):
        constraints.append(q_supply[i, t] <= cap)

# Solve the problem
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.SCS, verbose=True)

# Results
if problem.status == cp.OPTIMAL:
    print(f"Optimal cost: £{problem.value:.2f}")
    print("Generator dispatch (MW):")
    print(pd.DataFrame(q_supply.value, index=gen_ids, columns=node_demands.index))
else:
    print("No optimal solution found.")
