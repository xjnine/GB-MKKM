import numpy as np
from numpy.linalg import eigvals


# kcenter - center a kernel matrix
def kcenter(KH):
    KH = np.array(KH)
    n = KH.shape[1]  # 样本数

    if np.ndim(KH) == 2:
        # 计算每列元素的和
        D = np.sum(KH, axis=0) / n
        E = np.sum(D, axis=0) / n
        J = np.ones((n, 1)) * D
        KH = KH - J - J.T + E * np.ones((n, n))
        KH = 0.5 * (KH + KH.T)
    elif np.ndim(KH) == 3:
        numker = KH.shape[0]  # 核数
        for i in range(numker):
            # if is_positive_semidefinite(KH[i]):
            #     print("核矩阵" + str(i+1) + "是半正定的。")
            # else:
            #     print("核矩阵" + str(i+1) + "不是半正定的。")
            D = np.sum(KH[i], axis=0) / n
            E = np.sum(D, axis=0) / n
            J = np.ones((n, 1)) * D
            KH[i] = KH[i] - J - J.T + E * np.ones((n, n))
            KH[i] = 0.5 * (KH[i] + KH[i].T) + 1e-12 * np.eye(n)

    return KH


# 判断矩阵是不是半正定的
def is_positive_semidefinite(matrix):
    eigenvalues = eigvals(matrix)
    return all(e >= 0 for e in eigenvalues)
