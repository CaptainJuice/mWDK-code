# utils/data_loader.py

from scipy import io
import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import os
import warnings

warnings.filterwarnings('ignore')

def data2npy(dataset):
    """Convert raw graph data to .npy files."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_dir, "..", "datasets", "raw_data", dataset)
    base_path = os.path.abspath(base_path)

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset path not found: {base_path}")

    if dataset in ["amazon_cs",'amac']:
        npz_path = os.path.join(base_path, f"{dataset}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Data file not found: {npz_path}")
        data = np.load(npz_path)
        true_labels = data['labels']
        adj = data["adj_data"]
        attr = data["attr_data"]
        adj_mat = sp.csr_matrix((adj, data["adj_indices"], data["adj_indptr"]), shape=data['adj_shape']).todense()
        node_features = sp.csr_matrix((attr, data["attr_indices"], data["attr_indptr"]), shape=data['attr_shape']).todense()

    elif dataset in ["acm"]:
        mat_path = os.path.join(base_path, f"{dataset}.mat")
        data = sio.loadmat(mat_path)
        node_features = data['fea'].todense() if sp.issparse(data['fea']) else data['fea']
        adj_mat = (data['W'] > 0).astype(float)
        true_labels = (data['gnd'].T - 1)[0]

    elif dataset in ["BlogCatalog"]:
        edge_path = os.path.join(base_path, f"{dataset}.edge")
        feat_path = os.path.join(base_path, f"{dataset}.feature")
        label_path = os.path.join(base_path, f"{dataset}.label")

        edges = np.loadtxt(edge_path, dtype=int)
        node_features = np.loadtxt(feat_path)
        true_labels = np.loadtxt(label_path, dtype=int)
        N = node_features.shape[0]
        adj_mat = np.zeros((N, N))
        adj_mat[edges[:, 0], edges[:, 1]] = 1
        adj_mat[edges[:, 1], edges[:, 0]] = 1  # 无向

        # 读特征：每行一个节点

        # 读标签：每行一个整数

        # 保存为 mat 文件


        # node_features = data['fea'].todense() if sp.issparse(data['fea']) else data['fea']
        # adj_mat = (data['W'] > 0).astype(float)
        # true_labels = (data['gnd'].T - 1)[0]

    elif dataset == 'dblp':
        mat_path = os.path.join(base_path, "DBLP4057_GAT_with_idx.mat")
        data = sio.loadmat(mat_path)
        node_features = data['features'].todense() if sp.issparse(data['features']) else data['features']
        A, B, C = data['net_APTPA'], data['net_APCPA'], data['net_APA']
        adj_mat = (A + B + C > 0).astype(float)
        true_labels = np.argmax(data['label'], axis=1).squeeze()

    elif dataset == 'wiki':
        mat_path = os.path.join(base_path, f"{dataset}.mat")
        data = sio.loadmat(mat_path)
        node_features = data['fea'].todense() if sp.issparse(data['fea']) else data['fea']
        adj_mat = data['W'].todense() if sp.issparse(data['W']) else data['W']
        node_features = (node_features > 0).astype(float)
        true_labels = (data['gnd'].T - np.min(data['gnd'])).squeeze()

    elif dataset in ['cora', 'citeseer', 'pubmed']:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for name in names:
            file_path = os.path.join(base_path, f"ind.{dataset}.{name}")
            with open(file_path, 'rb') as rf:
                u = pkl.Unpickler(rf, encoding='latin1')
                objects.append(u.load())
        x, y, tx, ty, allx, ally, graph = objects

        test_idx_reorder = parse_index_file(os.path.join(base_path, f"ind.{dataset}.test.index"))
        test_idx_range = sorted(test_idx_reorder)

        if dataset == 'citeseer':
            from scipy.sparse import lil_matrix
            tx_ext = lil_matrix((len(test_idx_range), x.shape[1]))
            tx_ext[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_ext.tocsr()
            ty_ext = np.zeros((len(test_idx_range), y.shape[1]))
            ty_ext[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_ext

        features = sp.vstack([allx, tx]).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack([ally, ty])
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        node_features = features.todense()
        adj_mat = adj.todense()
        true_labels = np.argmax(labels, 1)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    adj_mat = adj_mat.astype(float)
    np.fill_diagonal(adj_mat, 0)
    node_features = node_features.astype(float)

    np.save(os.path.join(base_path, f"{dataset}_adj.npy"), adj_mat)
    np.save(os.path.join(base_path, f"{dataset}_feat.npy"), node_features)
    np.save(os.path.join(base_path, f"{dataset}_label.npy"), true_labels)

    return adj_mat, node_features, true_labels


def load_graph_data(path: str, dataset_name: str):
    """Load saved .npy files and return data with details."""
    base_path = os.path.join(path, 'raw_data', dataset_name)
    feat = np.load(f"{base_path}/{dataset_name}_feat.npy", allow_pickle=False)
    label = np.load(f"{base_path}/{dataset_name}_label.npy", allow_pickle=False)
    adj = np.load(f"{base_path}/{dataset_name}_adj.npy", allow_pickle=False)
    adj = (adj + adj.T) / 2
    details = (
        "++++++++++++++++++++++++++++++++\n"
        "--DETAILS OF GRAPH: [{}]--\n"
        "++++++++++++++++++++++++++++++++\n"
        "+ dataset name:   {}\n"
        "+ feature shape:  {}\n"
        "+ label shape:    {}\n"
        "+ adj shape:      {}\n"
        "+ edge num:       {}\n"
        "+ cluster num:   {}\n"
        "++++++++++++++++++++++++++++++++\n"
    ).format(
        dataset_name.upper(), dataset_name,
        feat.shape, label.shape, adj.shape,
        int(adj.sum() / 2),
        len(np.unique(label))
    )

    print(details)
    return feat, label, adj, details


def parse_index_file(filename):
    with open(filename, "r") as f:
        return [int(line.strip()) for line in f]


if __name__ == '__main__':
    datasets = ['cora']
    # 1.Convert raw data to .npy (can be skipped)
    # for dataset in datasets:
    #     data2npy(dataset)

    # 2. Load .npy and save as .mat
    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
    os.makedirs(input_dir, exist_ok=True)

    for dataset in datasets:
        try:
            features, labels, adj_mat, details = load_graph_data(input_dir, dataset)
            data = {
                'features': features,
                'labels': labels.reshape(-1, 1),
                'adj_mat': adj_mat,
                'details': details
            }
            io.savemat(f"{input_dir}/{dataset}.mat", data)
            print("Saved: {}.mat successfully.".format(dataset))
        except Exception as e:
            print("Failed to save {}: {}".format(dataset, e))
