import numpy as np
import sklearn.cluster as sc
import clustering_metric as cm
import scipy.sparse as sp
import logging
from IKBC import lkbc
from sklearn import preprocessing
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger('clustering')

def evaluate_clustering(true_labels, predict_labels):
    """Evaluate clustering performance."""
    _, new_pred = cm.translate(true_labels, predict_labels)
    results = cm.evaluationClusterModelFromLabel(true_labels, new_pred)
    acc, nmi, ari = results[0], results[1], results[-1]
    return acc, nmi, ari, new_pred


def sc_gaussian(dataset, embedding, num_of_clusters, true_labels, repetition=1):
    """Spectral clustering with RBF kernel (gamma tuning)."""
    best_nmi = -1
    best_labels = None
    best_acc = best_ari = 0.0
    best_gamma = 1.0
    gamma_list = [1, 10, 100]  # Candidate gamma values

    for gamma in gamma_list:
        local_acc, local_nmi, local_ari = [], [], []
        for _ in range(repetition):
            model = sc.SpectralClustering(n_clusters=num_of_clusters, gamma=gamma, random_state=42)
            pred = model.fit_predict(embedding)
            acc, nmi, ari, new_pred = evaluate_clustering(true_labels, pred)
            local_acc.append(acc)
            local_nmi.append(nmi)
            local_ari.append(ari)
        mean_nmi = np.mean(local_nmi)
        if mean_nmi > best_nmi:
            best_nmi = mean_nmi
            best_acc = np.mean(local_acc)
            best_ari = np.mean(local_ari)
            best_gamma = gamma
            best_labels = new_pred  # Store best prediction

    para = 'sc_gamma={:.4f}'.format(best_gamma)
    logger.info('@{}: ACC:{:.6f}  NMI:{:.6f}  ari_macro:{:.6f}'.format(dataset, best_acc, best_nmi, best_ari))
    return best_acc, best_nmi, best_ari, para, best_labels


def sc_linear(dataset, embedding, num_of_clusters, true_labels, repetition=1):
    """Linear spectral clustering."""
    embedding = preprocessing.normalize(embedding, norm='l2')
    best_nmi = -1
    best_labels = None
    local_acc, local_nmi, local_ari = [], [], []

    for _ in range(repetition):
        model = sc.SpectralClustering(n_clusters=num_of_clusters, affinity='linear')
        pred = model.fit_predict(embedding)
        acc, nmi, ari, new_pred = evaluate_clustering(true_labels, pred)
        local_acc.append(acc)
        local_nmi.append(nmi)
        local_ari.append(ari)
        if nmi > best_nmi:
            best_nmi = nmi
            best_labels = new_pred

    acc_mean = np.mean(local_acc)
    nmi_mean = np.mean(local_nmi)
    ari_mean = np.mean(local_ari)
    logger.info('@{}: ACC:{:.3f}  NMI:{:.3f}  ARI:{:.3f}'.format(dataset, acc_mean, nmi_mean, ari_mean))
    return acc_mean, nmi_mean, ari_mean, 'sc_linear', best_labels


def sc_semi(dataset, embedding, num_of_clusters, true_labels, repetition=1):
    """Spectral clustering with precomputed similarity matrix."""
    sim = embedding @ embedding.T
    best_nmi = -1
    best_labels = None
    local_acc, local_nmi, local_ari = [], [], []

    for _ in range(repetition):
        model = sc.SpectralClustering(
            n_clusters=num_of_clusters,
            affinity='precomputed',
            assign_labels='discretize',
            eigen_solver='arpack',
            random_state=66
        )
        pred = model.fit_predict(sim)
        acc, nmi, ari, new_pred = evaluate_clustering(true_labels, pred)
        local_acc.append(acc)
        local_nmi.append(nmi)
        local_ari.append(ari)
        if nmi > best_nmi:
            best_nmi = nmi
            best_labels = new_pred

    acc_mean = np.mean(local_acc)
    nmi_mean = np.mean(local_nmi)
    ari_mean = np.mean(local_ari)
    logger.info('@{}: ACC:{:.6f}  NMI:{:.6f}  ari_macro:{:.6f}'.format(dataset, acc_mean, nmi_mean, ari_mean))
    return acc_mean, nmi_mean, ari_mean, 'sc_semi', best_labels


def km(dataset, embedding, num_of_clusters, true_labels, repetition=1):
    """K-Means clustering."""
    best_nmi = -1
    best_labels = None
    local_acc, local_nmi, local_ari = [], [], []

    for _ in range(repetition):
        pred = sc.KMeans(n_clusters=num_of_clusters, random_state=42).fit_predict(embedding)
        acc, nmi, ari, new_pred = evaluate_clustering(true_labels, pred)
        local_acc.append(acc)
        local_nmi.append(nmi)
        local_ari.append(ari)
        if nmi > best_nmi:
            best_nmi = nmi
            best_labels = new_pred

    acc_mean = np.mean(local_acc)
    nmi_mean = np.mean(local_nmi)
    ari_mean = np.mean(local_ari)
    logger.info('@{}: ACC:{:.6f}  NMI:{:.6f}  ari_macro:{:.6f}'.format(dataset, acc_mean, nmi_mean, ari_mean))
    return acc_mean, nmi_mean, ari_mean, 'kms', best_labels


def sc_svd(dataset, embedding, num_of_clusters, true_labels, repetition=1):
    """SVD + KMeans clustering."""
    best_nmi = -1
    best_labels = None
    local_acc, local_nmi, local_ari = [], [], []

    # Compute top-k SVD components
    u, _, _ = sp.linalg.svds(embedding, k=num_of_clusters, which='LM')
    u = preprocessing.normalize(u, norm='l2')

    for _ in range(repetition):
        pred = sc.KMeans(n_clusters=num_of_clusters, random_state=42).fit_predict(u)
        acc, nmi, ari, new_pred = evaluate_clustering(true_labels, pred)
        local_acc.append(acc)
        local_nmi.append(nmi)
        local_ari.append(ari)
        if nmi > best_nmi:
            best_nmi = nmi
            best_labels = new_pred

    acc_mean = np.mean(local_acc)
    nmi_mean = np.mean(local_nmi)
    ari_mean = np.mean(local_ari)
    logger.info('@{}: ACC:{:.6f}  NMI:{:.6f}  ari_macro:{:.6f}'.format(dataset, acc_mean, nmi_mean, ari_mean))
    return acc_mean, nmi_mean, ari_mean, 'sc_svd', best_labels


def ikbc(dataset, embedding, num_of_clusters, true_labels, repetition=1):
    """Isolation Kernel-Based Clustering with tau search."""
    best_nmi = -1
    best_labels = None
    local_acc, local_nmi, local_ari = [], [], []
    best_acc = best_ari = 0.0

    for _ in range(repetition):
        acc_li, nmi_li, ari_li = [], [], []
        for t in range(2, 100, 2):
            tau = t * 0.01
            pred = lkbc(embedding, num_of_clusters, tau)
            acc, nmi, ari, new_pred = evaluate_clustering(true_labels, pred)
            acc_li.append(acc)
            nmi_li.append(nmi)
            ari_li.append(ari)
            if nmi > best_nmi:
                best_nmi = nmi
                best_acc = acc
                best_ari = ari
                best_labels = new_pred
        # Record max performance over tau
        local_acc.append(np.max(acc_li))
        local_nmi.append(np.max(nmi_li))
        local_ari.append(np.max(ari_li))

    acc_mean = np.mean(local_acc)
    nmi_mean = np.mean(local_nmi)
    ari_mean = np.mean(local_ari)
    logger.info('@{}: ACC:{:.6f}  NMI:{:.6f}  ari_macro:{:.6f}'.format(dataset, acc_mean, nmi_mean, ari_mean))
    return acc_mean, nmi_mean, ari_mean, 'ikbc', best_labels
