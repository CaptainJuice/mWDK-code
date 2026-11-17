import numpy as np
import scipy.sparse as sp
import networkx as nx

def normalize_adj(adj, typ='rw'):
    """Normalize adjacency matrix."""
    adj = sp.csr_matrix(adj, copy=True)
    if typ == 'sym':
        adj = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat = sp.diags(d_inv_sqrt)
        return d_mat @ adj @ d_mat
    elif typ == 'rw':
        rowsum = np.array(adj.sum(1)).flatten()
        rowsum[rowsum == 0] = 1.
        r_inv = 1.0 / rowsum
        r_mat_inv = sp.diags(r_inv)
        return r_mat_inv @ adj
    else:
        print("Unknown type: {}".format(typ))




# A demo of GSNN
def gsnn_adj_processing(A, r=0.5):
    G = nx.from_numpy_array(A)
    new_adj = np.zeros_like(A, dtype=float)
    for i in range(len(A)):
        neighbors = list(G.neighbors(i)) + [i]
        subg = G.subgraph(neighbors)
        sub_A = nx.adjacency_matrix(subg).toarray()
        for j in neighbors:
            if j == i:
                continue
            common = set(neighbors).intersection(set(list(G.neighbors(j)) + [j]))
            if len(common) > 1:
                c_idx = sorted(common)
                sub_idx = [list(neighbors).index(x) for x in c_idx]
                count = sub_A[np.ix_(sub_idx, sub_idx)].sum() / 2
                new_adj[i][j] = count / (len(common) * (len(common) - 1)) * (len(common)**r)
    new_adj += A
    W = new_adj + np.diag(np.sum(new_adj, axis=1))
    W = np.nan_to_num(W)
    D = np.diag(W.sum(1) + 1e-8)
    D_half = scipy.linalg.fractional_matrix_power(D, -0.5)
    return D_half @ W @ D_half


def GSNN(X, A, h, r=0.5):
    A_tilde = gsnn_adj_processing(A, r)
    P = np.linalg.matrix_power(A_tilde, h)
    return P @ X
