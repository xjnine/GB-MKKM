import numpy as np


# knorm - normalize a kernel matrix
# kn(x,y) = k(x,y) / sqrt(k(x,x) k(y,y))
def knorm(KH):
    n = KH.shape[1]  # 样本数
    numker = KH.shape[0]
    epsilon = 1e-6  # 对对角线元素小于等于0的进行处理
    if numker > 1:
        for i in range(numker):
            t1 = np.diag(KH[i])
            min = np.min(t1)
            if min < 0:
                min = -min
            if np.any(t1 <= epsilon):
                KH[i] = KH[i] + np.eye(n) * (min+epsilon)
            t1 = np.reshape(t1, (n, 1))
            t2 = t1.T
            t = t1 * t2
            diagonal = np.sqrt(t)
            KH[i] = KH[i] / diagonal
    else:
        KH = KH / np.sqrt(np.diag(KH) * np.diag(KH).T)
    return KH