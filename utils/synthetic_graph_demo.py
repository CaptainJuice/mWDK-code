import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.sparse import save_npz
import os
import warnings

warnings.filterwarnings('ignore')


# =========================
# 1. Gaussian Feature Generator
# =========================
def generate_gaussian_features(num_classes, nodes_per_class, dim, mu_list, sigma_list):
    """
    Generate node features from class-specific Gaussian distributions.

    Args:
        num_classes: int
        nodes_per_class: list[int], number of nodes in each class
        dim: feature dimension
        mu_list: list[float], mean offset for each class
        sigma_list: list[float], covariance scale for each class

    Returns:
        node_features (N x D), true_labels (N,)
    """
    node_features = []
    true_labels = []

    for c in range(num_classes):
        mu = mu_list[c] if isinstance(mu_list, list) else mu_list
        sigma = sigma_list[c] if isinstance(sigma_list, list) else sigma_list
        num_nodes = nodes_per_class[c]

        # Mean vector and covariance matrix
        mean = np.full(dim, mu)
        cov = np.eye(dim) * sigma

        # Sample from multivariate normal
        data = np.random.multivariate_normal(mean, cov, num_nodes)

        node_features.append(data)
        true_labels.extend([c] * num_nodes)

    node_features = np.vstack(node_features)
    true_labels = np.array(true_labels)

    # Normalize to [0, 1] per feature
    node_features -= node_features.min(axis=0, keepdims=True)
    max_val = node_features.max(axis=0, keepdims=True)
    max_val[max_val == 0] = 1  # avoid division by zero
    node_features /= max_val

    return node_features, true_labels


# =========================
# 2. Build SBM Graph with Targeted Degree
# =========================
def build_sbm_graph(nodes_per_class, inner_avg_deg, inter_avg_deg=None):
    """
    Build Stochastic Block Model graph with controlled intra/inter connectivity.

    Args:
        nodes_per_class: list[int]
        inner_avg_deg: float, target average degree within clusters
        inter_avg_deg: float, between clusters (default: ~10% of inner)

    Returns:
        adj_matrix (N x N), G (networkx graph)
    """
    n = sum(nodes_per_class)
    if inter_avg_deg is None:
        inter_avg_deg = max(0.5, inner_avg_deg * 0.1)  # default: weaker between

    sizes = nodes_per_class
    p_in = inner_avg_deg / (np.array(sizes) - 1 + 1e-8)  # avoid div by zero
    p_out = inter_avg_deg / (n - np.array(sizes) + 1e-8)

    # Use scalar or list depending on whether we want uniform or per-class
    pin = float(np.mean(p_in))
    pout = float(np.mean(p_out))

    G = nx.random_partition_graph(sizes, pin, pout)
    adj_mat = np.array(nx.adjacency_matrix(G).todense())

    return adj_mat, G


# =========================
# 3. Ensure Connectivity (fix isolated nodes)
# =========================
def ensure_connectivity(adj_mat, labels, max_attempts=10):
    """
    Fix isolated or low-degree nodes by adding edges.
    """
    deg = np.sum(adj_mat, axis=1)
    n = adj_mat.shape[0]

    for i in range(n):
        if deg[i] < 1:  # isolated
            for _ in range(max_attempts):
                j = random.randint(0, n - 1)
                if i != j:
                    adj_mat[i][j] = adj_mat[j][i] = 1
                    break
        elif deg[i] < 2:  # degree 1 → add one more edge
            for _ in range(3):
                j = random.randint(0, n - 1)
                if i != j:
                    adj_mat[i][j] = adj_mat[j][i] = 1
                    break

    return adj_mat


# =========================
# 4. Visualization Tools
# =========================
def plot_tsne(features, labels, title="t-SNE Visualization", save_path=None):
    """Plot t-SNE of high-dim features."""
    tsne = TSNE(n_components=2, perplexity=min(30, len(features) - 1),
                learning_rate='auto', init='pca', n_iter=1000, random_state=42)
    emb_2d = tsne.fit_transform(features)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=8, c=labels, cmap="rainbow", alpha=0.8)
    plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_adj(adj_mat, labels, title="Adjacency Matrix"):
    """Plot adjacency matrix with cluster boundaries."""
    plt.figure(figsize=(7, 6))
    plt.imshow(adj_mat, cmap="Blues")
    n = 0
    for size in np.cumsum(labels.bincount()):
        if n < len(labels):
            plt.axvline(x=size, color='red', linestyle='--', linewidth=1)
            plt.axhline(y=size, color='red', linestyle='--', linewidth=1)
        n += size
    plt.title(title)
    plt.colorbar()
    plt.show()


# =========================
# 5. Main: Generate Artificial Dataset
# =========================
def generate_artificial_dataset(
        name="synthetic",
        nodes_per_class=[1000, 1000],
        dim=100,
        mu_list=[0, 1],
        sigma_list=[10, 10],
        inner_avg_degree=6,
        inter_avg_degree=1.2,
        save_dir="datasets/"
):
    num_classes = len(nodes_per_class)
    total_nodes = sum(nodes_per_class)

    print(f"  Generating dataset '{name}'...")
    print(f"   Classes: {num_classes}, Nodes: {total_nodes}, Dim: {dim}")

    # Step 1: Features & Labels
    node_features, true_labels = generate_gaussian_features(
        num_classes=num_classes,
        nodes_per_class=nodes_per_class,
        dim=dim,
        mu_list=mu_list,
        sigma_list=sigma_list
    )

    # Step 2: Structure
    adj_mat, G = build_sbm_graph(
        nodes_per_class=nodes_per_class,
        inner_avg_deg=inner_avg_degree,
        inter_avg_deg=inter_avg_degree
    )
    adj_mat = ensure_connectivity(adj_mat, true_labels)

    # Step 3: Save paths
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(path, f"{name}_feat.npy"), node_features)
    np.save(os.path.join(path, f"{name}_label.npy"), true_labels)
    np.save(os.path.join(path, f"{name}_adj.npy"), adj_mat)

    # # Optional: sparse format for large graphs
    # save_npz(os.path.join(path, f"{name}_adj.npz"), nx.adjacency_matrix(G))

    # Step 4: Visualization
    # plot_tsne(node_features, true_labels,
    #           title=f"Feature Space ({name})",
    #           save_path=os.path.join(path, "tsne.png"))
    #
    # plot_adj(adj_mat, true_labels, title=f"Adjacency Matrix ({name})")

    # Step 5: Print stats
    deg = np.sum(adj_mat, axis=1)
    avg_deg = np.mean(deg)
    in_edges_0 = np.sum(adj_mat[:nodes_per_class[0], :nodes_per_class[0]]) / 2
    in_edges_1 = np.sum(adj_mat[nodes_per_class[0]:, nodes_per_class[0]:]) / 2
    cross_edges = np.sum(adj_mat[:nodes_per_class[0], nodes_per_class[0]:])

    density_ratio = cross_edges / (in_edges_0 + in_edges_1 + 1e-8)

    print(f"   Generated '{name}':")
    print(f"   Average degree: {avg_deg:.2f}")
    print(f"   Intra-cluster edges: C0={int(in_edges_0)}, C1={int(in_edges_1)}")
    print(f"   Inter-cluster edges: {int(cross_edges)}")
    print(f"   Density ratio (inter/intra): {density_ratio:.4f}")


# =========================
# Run Example
# =========================
if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)
    save_dir = os.path.join(project_root, "datasets", "raw_data", "artificial")
    name = "EEE"
    path = os.path.join(save_dir, name)

    # 自动创建目录
    generate_artificial_dataset(
        name=name,
        nodes_per_class=[1000, 1000],
        dim=100,
        mu_list=[0, 1],
        sigma_list=[10, 10],
        inner_avg_degree=6,
        inter_avg_degree=1.2,
        save_dir=path
    )
