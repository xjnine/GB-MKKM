import numpy as np


def sumKbeta(KH, beta):
    if not isinstance(KH, dict):
        ind = np.nonzero(beta)[0]  # 找到向量或矩阵中非零元素的索引
        n = int(KH[0].shape[0])
        Kaux = np.zeros((n, n))
        N = len(ind)
        for j in range(N):
            Kaux = Kaux + beta[ind[j]] * KH[ind[j]]
    # else:
    #     if beta.shape[0] > 1:
    #         beta = beta.T  # 转置操作
    #     if KH['data'].dtype == np.float32:
    #         Kaux = devectorize_single(np.dot(beta, KH['data'].T))
    #     else:
    #         Kaux = devectorize(np.dot(beta, KH['data'].T))
    return Kaux
