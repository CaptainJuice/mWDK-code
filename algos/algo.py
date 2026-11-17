import numpy as np
import random
from clustering import clustering_methods
import logging
from sklearn import preprocessing

logger = logging.getLogger(__name__)
class Algo:
    """
    Algorithm class for graph node embedding and clustering.
    Supports WL, WDK, and mWDK methods.
    """

    def __init__(self, dataset, algorithm, psi, t, h, repetition):
        self.dataset = dataset
        self.algorithm = algorithm
        self.psi = psi
        self.t = t
        self.h = h
        self.repetition = repetition

    def IK_fm_dot(self, X, psi, t):
        """An implementation of IK using anne."""
        n_samples = X.shape[0]
        onepoint_matrix = np.zeros((n_samples, int(t * psi)), dtype=int)
        x_index = np.arange(n_samples)

        for time in range(t):
            # Randomly sample psi points
            sample_idx = random.sample(range(n_samples), psi)
            sample = X[sample_idx]

            # Compute similarities
            point2sample = np.dot(X, sample.T)
            min_indices = np.argmax(point2sample, axis=1)  # closest sample index
            col_indices = min_indices + time * psi
            onepoint_matrix[x_index, col_indices] = 1

        return onepoint_matrix

    def IK_inne_fm(self, X, psi, t=100):
        """An implementation of IK using inne."""
        n_samples = X.shape[0]
        onepoint_matrix = np.zeros((n_samples, int(t * psi)), dtype=int)

        for time in range(t):
            sample_idx = random.sample(range(n_samples), psi)
            sample = X[sample_idx]

            # Compute squared distances: ||x - s||^2 = ||x||^2 + ||s||^2 - 2<x,s>
            xx = np.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
            ss = np.sum(sample ** 2, axis=1)             # (psi,)
            point2sample = xx + ss - 2 * np.dot(X, sample.T)  # (n, psi)

            # Compute radii from sample-to-sample distances
            sample2sample = point2sample[sample_idx]
            np.fill_diagonal(sample2sample, np.inf)
            radii = np.min(sample2sample, axis=1)

            # Find closest sample for each point
            min_indices = np.argmin(point2sample, axis=1)
            dist_values = point2sample[np.arange(n_samples), min_indices]

            # Only set bit if distance < radius of selected sample
            valid_mask = dist_values < radii[min_indices]
            col_indices = min_indices + time * psi
            onepoint_matrix[valid_mask, col_indices[valid_mask]] = 1

        return onepoint_matrix

    def create_adj_avg(self, adj_mat):
        """Normalize adjacency matrix."""
        adj = adj_mat - np.diag(np.diag(adj_mat))
        deg = np.sum(adj, axis=1)
        deg[deg == 0] = 1
        scaling = 0.5 / deg
        adj_normalized = adj * scaling[:, None]
        np.fill_diagonal(adj_normalized, 0.5)
        return adj_normalized

    def WL_noconcate(self, features, normalized_adj_mat):
        """One step ofWL without concatenation."""
        return np.dot(normalized_adj_mat, features)

    def get_embedding(self, adj_mat, node_features):
        """Generate embedding with each step recorded. Returns a list of embeddings."""
        logger.info("Generating embedding with {} (h={})".format(self.algorithm, self.h))
        normalized_adj = self.create_adj_avg(adj_mat)
        embedding = node_features.copy()
        embeddings = []

        if self.algorithm == 'WL':
            for i in range(self.h):
                embedding = self.WL_noconcate(embedding, normalized_adj)
                embedding_step = embedding.copy()
                embeddings.append(embedding_step)
                logger.info(f"Iteration {i + 1}/{self.h} [WL]: aggregation applied")

        elif self.algorithm == 'WDK':
            embedding = self.IK_fm_dot(embedding, self.psi, self.t)
            # embedding = preprocessing.normalize(embedding, norm='l2', axis=0)  # alternative
            for i in range(self.h):
                embedding = self.WL_noconcate(embedding, normalized_adj)
                embedding_step = embedding.copy()
                embeddings.append(embedding_step)
                logger.info(f"Iteration {i + 1}/{self.h} [WDK]: aggregation applied")

        elif self.algorithm == 'mWDK':
            for i in range(self.h):
                embedding = self.IK_fm_dot(embedding, self.psi, self.t)
                embedding = preprocessing.normalize(embedding, norm='l2', axis=0)  # alternative
                embedding = self.WL_noconcate(embedding, normalized_adj)
                embedding = preprocessing.normalize(embedding, norm='l2')  # test
                embedding_step = embedding.copy()
                embeddings.append(embedding_step)
                logger.info(f"Iteration {i + 1}/{self.h} [mWDK]:  aggregation applied")

        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        return embeddings

    def do_clustering(self, embeddings, num_of_clusters, true_labels, repetition=10):
        """
        Perform clustering on each embedding and return the best result across all steps.

        Args:
            embeddings: list of np.array, each is an embedding at step h
            num_of_clusters: int
            true_labels: ground truth labels
            repetition: how many times to repeat clustering per step (for stability)

        Returns:
            best_acc, best_nmi, best_ari, best_para, best_pred
        """
        from clustering.clustering_methods import sc_linear
        logger = logging.getLogger('clustering')

        best_acc = 0.0
        best_nmi = 0.0
        best_ari = 0.0
        best_pred = None
        best_iter = 0
        best_para = ''

        logger.info("Starting  clustering for %s", self.algorithm)

        for step_idx, embedding in enumerate(embeddings):
            step_nmi_list = []
            step_acc_list = []
            step_ari_list = []

            for rep in range(repetition):
                acc, nmi, ari, para, pred = sc_linear(
                    dataset=self.dataset,
                    embedding=embedding,
                    num_of_clusters=num_of_clusters,
                    true_labels=true_labels,
                    repetition=1
                )
                step_acc_list.append(acc)
                step_nmi_list.append(nmi)
                step_ari_list.append(ari)

            mean_acc = np.mean(step_acc_list)
            mean_nmi = np.mean(step_nmi_list)
            mean_ari = np.mean(step_ari_list)

            logger.info('@%s: Clustering %d -> ACC=%.4f, NMI=%.4f, ARI=%.4f',
                        self.dataset, step_idx + 1, mean_acc, mean_nmi, mean_ari)

            if mean_nmi > best_nmi:
                best_nmi = mean_nmi
                best_acc = mean_acc
                best_ari = mean_ari
                best_pred = pred.copy()
                best_iter = step_idx + 1
                best_para = f"step{step_idx + 1}_avg"

        logger.info(' @%s: Clustering result at iteration %d -> ACC=%.6f, NMI=%.6f, ARI=%.6f',
                    self.dataset, best_iter, best_acc, best_nmi, best_ari)

        return best_acc, best_nmi, best_ari, best_para, best_pred




