import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the data
edges_data = pd.read_csv("network_edges.csv")  # Update with correct path
generator_data = pd.read_csv("provided_material/generators.csv")  # Update with correct path
node_demands = pd.read_csv("provided_material/hourlydemandbynode.csv", index_col=0)

# Create the graph
G = nx.Graph()

# Add edges to the graph
for _, row in edges_data.iterrows():
    G.add_edge(row['FROM_NODE'], row['TO_NODE'])

# Identify nodes
generator_nodes = set(generator_data['NODE'])  # Ensure 'NODE' is the correct column name
demand_nodes = set(map(int, node_demands.columns))  # Ensure columns are integer node IDs

# Debugging: Verify Demand and Generator Nodes
print("Demand Nodes Set:", demand_nodes)
print("Generator Nodes Set:", generator_nodes)

# Check if nodes are in the graph
print("Demand Nodes in Graph:", [node for node in demand_nodes if node in G.nodes()])
print("Generator Nodes in Graph:", [node for node in generator_nodes if node in G.nodes()])

# Verify node color assignment
node_colors = []
for node in G.nodes():
    if node in demand_nodes:
        node_colors.append("blue")  # Demand nodes
    elif node in generator_nodes:
        node_colors.append("green")  # Generator nodes
    else:
        node_colors.append("black")  # Intermediate nodes
    print(f"Node: {node}, Assigned Color: {node_colors[-1]}")  # Debug output

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
plt.title("Network Graph with Debugged Node Colors")
plt.show()

