from sklearn import metrics
from munkres import Munkres
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def translate(true_label, predict_label):
    """
    Find the best mapping between true labels and predicted labels using Hungarian algorithm.

    Args:
        true_label: array-like, shape (n_samples,)
        predict_label: array-like, shape (n_samples,)

    Returns:
        translate: list of mapped cluster representatives
        mapping_predict: remapped predictions to match true labels
    """
    l1 = np.unique(true_label)
    num_class1 = len(l1)

    l2 = np.unique(predict_label)
    num_class2 = len(l2)

    # Build cost matrix: rows=true classes, cols=predicted classes
    cost = np.zeros((num_class1, num_class2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [idx for idx, e1 in enumerate(true_label) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [idx for idx in mps if predict_label[idx] == c2]
            cost[i][j] = len(mps_d)

    # Apply Munkres algorithm to maximize overlap (minimize negative cost)
    m = Munkres()
    indices = m.compute((-cost).tolist())

    mapping_predict = np.zeros(len(predict_label), dtype=int)
    translate_list = []

    for i, c in enumerate(l1):
        try:
            c2_idx = indices[i][1]
            c2 = l2[c2_idx]
        except IndexError:
            return predict_label.copy(), predict_label.copy()

        ai = [ind for ind, elm in enumerate(predict_label) if elm == c2]
        translate_list.append(c2)
        mapping_predict[ai] = int(c)

    return translate_list, mapping_predict


def evaluationClusterModelFromLabel(true_label, predict_label):
    """
    Evaluate clustering performance with multiple metrics.

    Returns:
        acc, nmi, f1_macro, precision_macro, recall_macro,
        f1_micro, precision_micro, recall_micro, adjscore
    """
    acc = metrics.accuracy_score(true_label, predict_label)
    nmi = metrics.normalized_mutual_info_score(true_label, predict_label)
    f1_macro = metrics.f1_score(true_label, predict_label, average='macro')
    precision_macro = metrics.precision_score(true_label, predict_label, average='macro')
    recall_macro = metrics.recall_score(true_label, predict_label, average='macro')
    f1_micro = metrics.f1_score(true_label, predict_label, average='micro')
    precision_micro = metrics.precision_score(true_label, predict_label, average='micro')
    recall_micro = metrics.recall_score(true_label, predict_label, average='micro')
    adjscore = metrics.adjusted_rand_score(true_label, predict_label)

    return (
        acc, nmi, f1_macro, precision_macro, recall_macro,
        f1_micro, precision_micro, recall_micro, adjscore
    )


def plot(X, fig, colors, size, true_labels):
    """
    Plot embedded points with color-coded true labels.
    """
    ax = fig.add_subplot(1, 1, 1)
    for i, point in enumerate(X):
        ax.scatter(point[0], point[1], s=size, c=colors[true_labels[i]])


def plotClusters(hidden_emb, true_labels):
    """
    Perform t-SNE dimensionality reduction and plot clusters.
    """
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(hidden_emb)

    fig = plt.figure(figsize=(8, 6))
    plot(X_tsne, fig, ['red', 'green', 'blue', 'brown', 'purple',
                       'yellow', 'pink', 'orange'], 4, true_labels)
    plt.axis('off')
    plt.tight_layout()
    fig.savefig("plot.png", dpi=300)
    plt.show()
    print("Finished plotting, saved to plot.png")
