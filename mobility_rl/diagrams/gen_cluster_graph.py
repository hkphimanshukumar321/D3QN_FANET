import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull
import os

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

np.random.seed(42)
fig, ax = plt.subplots(figsize=(4.5, 3.5))

# Define centers and create clusters
centers = np.array([[2, 8], [8, 7], [5, 2]])
colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

nodes_per_cluster = 6
pos = {}
node_cluster = {}
ch_nodes = [0, nodes_per_cluster, nodes_per_cluster*2]

# Generate positions
for i in range(3):
    center = centers[i]
    for j in range(nodes_per_cluster):
        node_id = i * nodes_per_cluster + j
        if j == 0:
            pos[node_id] = center # CH exactly at center
        else:
            pos[node_id] = center + np.random.randn(2) * 1.5
        node_cluster[node_id] = i

G = nx.Graph()
for node_id in pos.keys():
    G.add_node(node_id, pos=pos[node_id])

# Intra-cluster edges (CM to CH)
intra_edges = []
for i in range(3):
    ch = ch_nodes[i]
    for j in range(1, nodes_per_cluster):
        cm = i * nodes_per_cluster + j
        intra_edges.append((ch, cm))

# Inter-cluster edges (interference graph G_t)
inter_edges = [(ch_nodes[0], ch_nodes[1]), (ch_nodes[1], ch_nodes[2]), (ch_nodes[2], ch_nodes[0])]

# Draw shaded regions
for i in range(3):
    cluster_pts = np.array([pos[n] for n, c in node_cluster.items() if c == i])
    hull = ConvexHull(cluster_pts)
    hull_pts = cluster_pts[hull.vertices]
    poly = plt.Polygon(hull_pts, fill=True, color=colors[i], alpha=0.15, edgecolor="none")
    ax.add_patch(poly)
    
    # Boundary
    hull_pts_closed = np.vstack((hull_pts, hull_pts[0]))
    ax.plot(hull_pts_closed[:,0], hull_pts_closed[:,1], color=colors[i], linestyle="--", linewidth=1, alpha=0.5)

# Draw intra-cluster edges
nx.draw_networkx_edges(G, pos, edgelist=intra_edges, width=1.0, edge_color="#7f7f7f", style="solid", ax=ax)

# Draw inter-cluster edges (GAT interference)
nx.draw_networkx_edges(G, pos, edgelist=inter_edges, width=1.5, edge_color="#d62728", style="dashed", ax=ax)

# Draw CM nodes
for i in range(3):
    cms = [n for n, c in node_cluster.items() if c == i and n not in ch_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=cms, node_size=60, node_color=colors[i], edgecolors="white", ax=ax)

# Draw CH nodes
for i in range(3):
    nx.draw_networkx_nodes(G, pos, nodelist=[ch_nodes[i]], node_size=200, node_shape="*", node_color=colors[i], edgecolors="#333333", ax=ax)

# Labels for Clusters and CHs
for i, name in enumerate(["$C_1$", "$C_2$", "$C_3$"]):
    ax.text(centers[i,0], centers[i,1] + 2.0, name, fontsize=12, fontweight="bold", color=colors[i], ha="center", va="center")
    ax.text(centers[i,0]+0.3, centers[i,1]-0.4, f"$CH_{i+1}$", fontsize=9, fontweight="bold", color="#333")

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='*', color='w', label='Cluster Head (CH)', markerfacecolor='#7f7f7f', markeredgecolor='#333333', markersize=12),
    Line2D([0], [0], marker='o', color='w', label='Cluster Member (CM)', markerfacecolor='#7f7f7f', markeredgecolor='white', markersize=8),
    Line2D([0], [0], color='#7f7f7f', lw=1.5, label='Intra-cluster Link'),
    Line2D([0], [0], color='#d62728', lw=1.5, linestyle='--', label='Inter-cluster Interference Edge ($\mathcal{E}_t$)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.9, edgecolor="#ccc")

ax.axis("off")
out_path = "alternate_tj_latex_template_ap/figures/early_clustering_graph.png"
plt.savefig(out_path)
plt.savefig(out_path.replace(".png", ".pdf"))
print("Generated cluster graph.")
