import math
import scipy.io
# import h5py
import numpy as np
import time
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from functions.knorm import knorm
from functions.mycomFun import mycomFun
from functions.generateNeighborhood import generateNeighborhood
from GBMKKM.localizedGBMKKM import localizedGBMKKM
from ClusteringEvaluation.myMetric import myMetric, metric
from gbutils.splitGBs import splitGBs
from gbutils.similarity import calEuclidDistanceMatrix


def loadfile(path, type):
    if type == "csv":
        datas = pd.read_csv(path, header=None)
        data = datas.values[:, 2:]
        trueLabel = datas.values[:, 1]

    elif type == "mat":
        mat_data = scipy.io.loadmat(path)
        data = mat_data['fea'][:, :]
        trueLabel = mat_data['gt'][:, :]
        trueLabel = [element for row in trueLabel for element in row]
    return data, trueLabel


def normalized(data, dimension):
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))  # 数据缩放
    value = data[:, :]
    if len(value[0]) > dimension:
        pca = PCA(n_components=dimension)
        value = pca.fit_transform(value)
    value = min_max_scaler.fit_transform(value)
    return value

# 10个核：高斯核
def load_KH_Y_v3(path):
    # data, label = loadfile(path, "mat")

    data, label = load_iris().data, load_iris().target
    # csvdata = pd.read_csv(path, header=None)
    # data, label = csvdata.values[:, 1:], csvdata.values[:, 0]
    data = normalized(data, len(data[0]))  # 归一化(不降维)

    datalist = data

    # 使用粒球中心代替数据
    print("开始生成粒球...")

    gb_list = splitGBs(data)  # 加权分裂

    print("粒球数量：" + str(len(gb_list)))

    gb_center_list = [gb.center for gb in gb_list]

    data = np.vstack(gb_center_list)

    num = len(gb_list)

    S = calEuclidDistanceMatrix(gb_list)  # 计算粒球距离矩阵

    average_dist = np.sum(S) / (((num * num - num) / 2) * 2)  # 平均球间距离

    ## End

    # avg_cov_matrices = np.cov(data, rowvar=False)
    # c = np.sqrt(np.trace(avg_cov_matrices))
    # RBF_Kernel_comfient = [0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]
    # sigmalist = [i * c for i in RBF_Kernel_comfient]
    Do = np.max(S)  # 粒球间最大距离
    RBF_Kernel_comfient = [0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]
    sigmalist = [i * Do for i in RBF_Kernel_comfient]

    # KH = np.zeros(len(sigmalist))  # 包含多个核矩阵 核数 * 样本数 * 样本数
    KH = []

    # overlap_list = []
    # flag = True
    print("-----------------------开始构造核矩阵----------------------")
    for i in range(len(sigmalist)):
        KH_item = np.zeros((num, num))  # 初始化每个核矩阵
        for j in range(num):
            # overlap = 0
            for k in range(j, num):

                # 如果球j与球k相交
                if S[j][k] <= gb_list[j].radius + gb_list[k].radius:
                    # overlap += 1
                    # KH_item[j, k] = gaussian_kernel(data[j], data[k], sigmalist[i])
                    nearest_point, nearest_distance = find_nearest_point(gb_list[j],
                                                                         gb_list[k])  # 找出第k个粒球中 离 粒球j中心最近的点 来代替 粒球k
                    KH_item[j, k] = gaussian_kernel(data[j], nearest_point, sigmalist[i])

                # 如果球i与球j不相交
                else:
                    if S[j][k] > average_dist:
                        KH_item[j, k] = gaussian_kernel(data[j], data[k], sigmalist[i])
                    else:
                        nearest_point, nearest_distance = find_nearest_point(gb_list[j],
                                                                             gb_list[k])  # 找出第k个粒球中 离 粒球j中心最近的点 来代替 粒球k
                        KH_item[j, k] = gaussian_kernel(data[j], nearest_point, sigmalist[i])

                KH_item[k, j] = KH_item[j, k]
            # overlap_list.append(overlap)

        # if flag:
        #     flag = False
        #     print("粒球重叠次数：", overlap_list)
        #     print("平均重叠次数("+ str(len(overlap_list)) + ")：", np.mean(overlap_list))

        KH.append(KH_item)
        print("-----构造第" + str(i + 1) + "个核矩阵完成-------")
    print("-----------------------构造核矩阵结束----------------------")
    KH = np.array(KH)
    return KH, label, gb_list, datalist


# 7个核：高斯核 多项式核 余弦核
def load_KH_Y_v3_2(path):
    data, label = loadfile(path, "csv")

    data = normalized(data, len(data[0]))  # 归一化(不降维)

    datalist = data

    # 使用粒球中心代替数据
    print("开始生成粒球...")

    gb_list = splitGBs(data)  # 分裂

    print("粒球数量：" + str(len(gb_list)))

    gb_center_list = [gb.center for gb in gb_list]

    data = np.vstack(gb_center_list)

    num = len(gb_list)

    S = calEuclidDistanceMatrix(gb_list)  # 计算粒球距离矩阵

    average_dist = np.sum(S) / (((num * num - num) / 2) * 2)  # 平均球间距离

    # 高斯核参数
    Do = np.max(S)  # 粒球间最大距离
    RBF_Kernel_comfient = [0.25, 0.5, 1, 2, 3, 4]
    sigmalist = [i * Do for i in RBF_Kernel_comfient]

    KH = np.zeros((len(sigmalist) + 1, num, num))  # 包含多个核矩阵 核数 * 样本数 * 样本数
    for i in range(num):
        for j in range(i, num):
            point_1 = data[i]
            # 如果球j与球k相交
            if S[i][j] <= gb_list[i].radius + gb_list[j].radius:
                nearest_point, nearest_distance = find_nearest_point(gb_list[i],
                                                                     gb_list[j])  # 找出第j个粒球中 离 粒球i中心最近的点 来代替 粒球j
                point_2 = nearest_point

            # 如果球i与球j不相交
            else:
                if S[i][j] > average_dist:
                    point_2 = data[j]
                else:
                    nearest_point, nearest_distance = find_nearest_point(gb_list[i],
                                                                         gb_list[j])  # 找出第j个粒球中 离 粒球i中心最近的点 来代替 粒球j
                    point_2 = nearest_point

            # 高斯核核矩阵
            for index in range(0, len(sigmalist)):
                KH[index, i, j] = gaussian_kernel(point_1, point_2, sigmalist[index])
                KH[index, j, i] = KH[index, i, j]
            index += 1
            # 余弦核矩阵
            KH[index, i, j] = cosine_kernel(point_1, point_2)
            KH[index, j, i] = KH[index, i, j]

    KH = np.array(KH)
    return KH, label, gb_list, datalist


def find_nearest_point(gb1, gb2):
    query_point = np.array(gb1.center)
    points = np.array(np.vstack((gb2.data, gb2.center)))
    distances = np.linalg.norm(points - query_point, axis=1)
    nearest_index = np.argmin(distances)
    nearest_point = points[nearest_index]
    nearest_distance = distances[nearest_index]
    return nearest_point, nearest_distance


# 高斯核
def gaussian_kernel(x1, x2, sigma):
    """
    计算两个样本之间的高斯核值

    参数：
    x1: 第一个样本，numpy数组
    x2: 第二个样本，numpy数组
    sigma: 高斯核的标准差

    返回：
    高斯核值
    """
    # distance = np.linalg.norm(x1 - x2)
    distance = np.linalg.norm((x1 - x2))
    kernel_value = np.exp(-0.5 * (distance / sigma) ** 2)
    return kernel_value


# 余弦核
def cosine_kernel(x1, x2):
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    return np.dot(x1, x2) / (norm_x1 * norm_x2)


# 初始化参数
def init(KH, Y):
    num_class = len(np.unique(Y))
    # Y[Y < 1] = num_class  # 将小于1的值替换为num_class

    numker = KH.shape[0]
    num = KH.shape[1]

    #  Setting some numerical parameters
    options = {}
    options['seuildiffsigma'] = 1e-5  # stopping criterion for weight variation
    options['goldensearch_deltmax'] = 1e-1  # initial precision of golden section search
    options['numericalprecision'] = 1e-10  # numerical precision weights below this value are set to zero

    # some algorithms paramaters
    options[
        'firstbasevariable'] = 'first'  # tie breaking method for choosing the base variable in the reduced gradient method
    options['nbitermax'] = 500  # maximal number of iteration
    options['seuil'] = 0  # forcing to zero weights lower than this value, for iterations lower than this one
    options['seuilitermax'] = 10
    options['miniter'] = 0  # minimal number of iterations
    options['threshold'] = 1e-4
    options['goldensearchmax'] = 1e-3
    options['seuildiffsigma'] = 5e-3  # stopping criterion for weight variation
    return options


# 算法
def GBMKKM(path):
    KH, Y, gb_list, datalist = load_KH_Y_v3_2(path)
    options = init(KH, Y)
    numker = KH.shape[0]
    num = KH.shape[1]
    num_class = len(np.unique(Y))

    KH_normalized = knorm(KH)  # 归一化

    # GBMKKM
    # start = time.time()
    sigma = np.ones((numker, 1)) / numker
    avgKer = mycomFun(KH_normalized, sigma)

    tau = 1
    numSel = round(tau * num)

    NS = generateNeighborhood(avgKer, numSel)

    NS = NS.T
    A = np.zeros((num, num))
    for i in range(num):
        indices = NS[:, i]
        indices = np.reshape(indices, (len(indices), 1))

        A[[indices[:, None]], indices] += 1
    H_normalized, sigma, obj = localizedGBMKKM(KH_normalized, num_class, A, options)

    ari_arr = []
    nmi_arr = []
    acc_arr = []
    MAXITER = 10
    for i in range(MAXITER):
        gb_labels = myMetric(H_normalized, Y, num_class)  # H_normalized执行k-means

        clusters, gb_dict = get_clusters(gb_list, gb_labels)

        ari, nmi, acc, labels_pre, labels_true, plot_data = evaluate(gb_dict, datalist, Y)
        ari_arr.append(ari)
        nmi_arr.append(nmi)
        acc_arr.append(acc)

    ari = np.mean(ari_arr)
    nmi = np.mean(nmi_arr)
    acc = np.mean(acc_arr)

    # END GBMKKM
    print("[ari,nmi,acc]：", "[" + str(ari) + "," + str(nmi) + "," + str(acc) + "]")
    print("----------------------------------------")
    print("sigma:", sigma)
    print("----------------------------------------")


def get_clusters(gb_list, gb_list_labels):
    clusters = {}
    gb_dict = {}
    for i in range(0, len(gb_list)):
        gb_list[i].label = gb_list_labels[i]
        gb_dict[i] = gb_list[i]
        if gb_list_labels[i] in clusters.keys():
            clusters[gb_list_labels[i]] = np.append(clusters[gb_list_labels[i]], gb_list[i].data, axis=0)
        else:
            clusters[gb_list_labels[i]] = gb_list[i].data
    return clusters, gb_dict


def evaluate(gb_dict, data, trueLabel):
    trueLabel = np.array(trueLabel).reshape(-1)
    data_list = data.tolist()
    label_set = set()

    # (zt)计算ARI
    labels_true = []
    labels_pre = []
    plot_data = []

    for gb in gb_dict.values():
        if gb.label not in label_set:
            label_set.add(gb.label)

        for p in gb.data.tolist():
            # labels_true.append(int(float(self.trueLabel[data_list.index(p)])))
            labels_true.append(trueLabel[data_list.index(p)])
            labels_pre.append(gb.label)
            plot_data.append(p)

    ari, nmi, acc = metric(labels_true, labels_pre)

    return ari, nmi, acc, labels_pre, labels_true, plot_data


if __name__ == "__main__":
    path = "./dataset/" + "wdbc.data"

    start = time.time()

    GBMKKM(path)

    end = time.time()

    costTime = (end - start)

    print("time(s):", costTime)
