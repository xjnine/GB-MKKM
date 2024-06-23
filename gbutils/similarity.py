import numpy as np


# 谱：sqrt_flag=False
def euclidDistance(gb1, gb2, sqrt_flag=True):
    res = ((gb1.center - gb2.center) ** 2).sum(axis=0)
    if sqrt_flag:
        res = np.sqrt(res)
    return res


def calEuclidDistanceMatrix(gb_list):
    gb_list = np.array(gb_list)
    S = np.zeros((len(gb_list), len(gb_list)))
    for i in range(len(gb_list)):
        for j in range(i + 1, len(gb_list)):
            S[i][j] = 1.0 * euclidDistance(gb_list[i], gb_list[j])
            S[j][i] = S[i][j]
    return S
