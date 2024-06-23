import numpy as np


def mycomFun(KH, sigma):
    m = KH.shape[0]  # 核数
    n = KH.shape[1]  # 样本数
    cF = np.zeros((n, n))
    for i in range(m):
        cF += KH[i] * sigma[i]
    return cF
