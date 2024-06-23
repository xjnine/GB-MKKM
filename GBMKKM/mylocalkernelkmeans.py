import numpy as np
from scipy.sparse.linalg import eigs


def mylocalkernelkmeans(Kmatrix, A, numclass):
    opt = {'disp': 0}
    K0 = np.multiply(A, Kmatrix)
    K0 = (K0 + K0.T) / 2
    # _, H = eigs(K0, numclass, which='LA', **opt)  # 计算稀疏矩阵的特征值和特征向量
    _, H = eigs(K0, numclass, which='LM')  # 计算稀疏矩阵的特征值和特征向量
    obj = np.trace(H.T @ K0 @ H)
    H = np.real(H)
    obj = np.real(obj)
    return H, obj
