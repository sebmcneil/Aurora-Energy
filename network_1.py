import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Create the graph
G = nx.Graph()

# Load data
network_edges = pd.read_csv("network_edges.csv")  # Replace with the actual file path
node_demands = pd.read_csv("hourlydemandbynode.csv", index_col=0)  # Replace with the actual file path
generator_data = pd.read_csv("generators.csv", index_col=0)  # Replace with the actual file path

# Optional: Strip prefixes from node IDs if needed
network_edges["FROM_NODE"] = network_edges["FROM_NODE"].str.replace("N_", "").astype(int)
network_edges["TO_NODE"] = network_edges["TO_NODE"].str.replace("N_", "").astype(int)

# Add edges to the graph
for _, row in network_edges.iterrows():
    G.add_edge(row["FROM_NODE"], row["TO_NODE"])

# Prepare demand and generator nodes
demand_nodes = set(map(int, node_demands.columns))  # Ensure integers
generator_nodes = set(map(int, generator_data["NODE"]))  # Ensure integers

# Debug: Check node existence
print("Demand Nodes in Graph:", [node for node in demand_nodes if node in G.nodes()])
print("Generator Nodes in Graph:", [node for node in generator_nodes if node in G.nodes()])

# Add missing nodes (if any)
for node in demand_nodes.union(generator_nodes):
    if node not in G.nodes():
        G.add_node(node)

# Assign colors to nodes
node_colors = []
for node in G.nodes():
    if node in demand_nodes:
        node_colors.append("blue")  # Demand nodes
    elif node in generator_nodes:
        node_colors.append("green")  # Generator nodes
    else:
        node_colors.append("black")  # Intermediate nodes

# Plot the graph
plt.figure(figsize=(12, 12))
nx.draw(
    G,
    pos=nx.spring_layout(G, seed=42),
    node_color=node_colors,
    with_labels=True,
    node_size=100,
    font_size=8,
)
plt.title("Network Graph with Demand and Generator Nodes Highlighted")
plt.show()
