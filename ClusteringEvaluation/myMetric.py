import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment as linear_assignment

np.random.seed()


def myMetric(U, Y, numclass):
    Y = np.array(Y).reshape(-1)
    U_normalized = U / np.sqrt(np.sum(U ** 2, axis=1, keepdims=True))  # 归一化
    maxIter = 20

    res_ari = np.zeros(maxIter)
    res_nmi = np.zeros(maxIter)
    res_acc = np.zeros(maxIter)
    # res4 = np.zeros(maxIter)
    # for i in range(maxIter):
    #     kmeans = KMeans(n_clusters=numclass, max_iter=100, n_init=10, random_state=None)
    #     pre_labels = kmeans.fit_predict(U_normalized)
    #     ari, nmi, acc = metric(Y, pre_labels)
    #     res_ari[i] = ari
    #     res_nmi[i] = nmi
    #     res_acc[i] = acc
    # mean_ari = np.mean(res_ari)
    # mean_nmi = np.mean(res_nmi)
    # mean_acc = np.mean(res_acc)
    # return mean_ari, mean_nmi, mean_acc, pre_labels
    kmeans = KMeans(n_clusters=numclass, max_iter=100, n_init=10, random_state=None)

    pre_labels = kmeans.fit_predict(U_normalized)

    return pre_labels


def metric(true_labels, pre_labels):
    ari = adjusted_rand_score(true_labels, pre_labels)
    nmi = normalized_mutual_info_score(true_labels, pre_labels)
    acc = acc_score(true_labels, pre_labels)
    return ari, nmi, acc


def acc_score(y_true, y_pred):
    """
        Calculate clustering accuracy. Require scikit-learn installed
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """
    """将非整型标签映射为整型标签"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if isinstance(y_true[0], str):
        y_true_map = []
        map_dict = {}
        l = 0
        for i in y_true:
            if i not in map_dict:
                l += 1
            y_true_map.append(map_dict.setdefault(i, l))
        y_true = np.array(y_true_map)
    """end"""

    """将非整型标签映射为整型标签"""
    if isinstance(y_pred[0], str):
        y_pred_map = []
        mapred_dict = {}
        ll = 0
        for i in y_pred:
            if i not in mapred_dict:
                ll += 1
            y_pred_map.append(mapred_dict.setdefault(i, ll))
        y_pred = np.array(y_pred_map)
    """end"""
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    ind = np.array(ind).T
    accuracy = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    return accuracy
