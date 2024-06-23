import numpy as np


def localGBMKKMGrad(KH, A, Hstar, Sigma):
    d = KH.shape[0]  # 核数
    grad = np.zeros((d, 1))
    for k in range(d):
        temp = np.trace(Hstar.T @ (KH[k] * A) @ Hstar)
        grad[k] = 2 * Sigma[k] * temp
    return grad
