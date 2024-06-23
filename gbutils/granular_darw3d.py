from sklearn.preprocessing import StandardScaler, MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D, axes3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import k_means


class GB:
    def __init__(self, data, label):  # Data is labeled data, the penultimate column is label, and the last column is index
        self.data = data
        self.center = self.data.mean(0)# According to the calculation of row direction, the mean value of all the numbers in each column (that is, the center of the pellet) is obtained
        self.radius = self.get_radius()
        self.flag = 0
        self.label = label
        self.num = len(data)
        self.out = 0
        self.size = 1

    def get_radius(self):
        return max(((self.data - self.center) ** 2).sum(axis=1) ** 0.5)


class UF:
    def __init__(self,len):
        self.parent = [0]*len
        self.size = [0]*len
        self.count = len

        for i in range(0,len):
            self.parent[i] = i
            self.size[i] = 1

    def find(self, x):
        while (self.parent[x] != x):
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if (rootP == rootQ):
            return
        if self.size[rootP] > self.size[rootQ]:
            self.parent[rootQ] = rootP
            self.size[rootP] += self.size[rootQ]
        else:
            self.parent[rootP] = rootQ
            self.size[rootQ] += self.size[rootP]
        self.count = self.count - 1


    def connected(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        return rootP == rootQ

    def count(self):
        return self.count

# 绘制数据点2d图
def plot_dot(data):
    color = {0: '#FF0000', 1: '#000000', 2: '#0000FF'}
    plt.figure(figsize=(6, 6))
    plt.axis([-2, 2, -2, 2])
    plt.plot(data[:, 0], data[:, 1], '.', color=color[0], markersize=3)


#绘制数据点3d图
def plot_3d_dot(data,ax):
 
    ax.scatter(data[:, 0], data[:, 1],0)

# 画出粒球的圆
def draw_ball(gb_list):
    for data in gb_list:
        if len(data) >1 :
            center = data.mean(0)
            # radius = np.mean((((data - center) ** 2).sum(axis=1) ** 0.5))
            radius = np.max((((data - center) ** 2).sum(axis=1) ** 0.5))
            # print(center, radius)
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            plt.plot(x, y, '#000000', linewidth=0.8)
        else:
            # print(data)
            plt.plot(data[0][0], data[0][1],marker = '*',color= '#0000FF', markersize=3)
    plt.show()



#绘制3d粒球
def draw_3d_ball(gb_list,ax):
    color = {
        0: '#707afa',
        1: '#ffe135',
        2: '#16ccd0',
        3: '#ed7231',
        4: '#0081cf',
        5: '#afbed1',
        6: '#bc0227',
        7: '#d4e7bd',
        8: '#f8d7aa',
        9: '#fecf45',
        10: '#f1f1b8',
        11: '#b8f1ed',
        12: '#ef5767',
        13: '#e7bdca',
        14: '#8e7dfa',
        15: '#d9d9fc',
        16: '#2cfa41',
        17: '#e96d29',
        18: '#7f722f',
        19: '#bd57fa',
        20: '#e4f788',
        21: '#fb8e94',
        22: '#b8d38f',
        23: '#e3a04f',
        24: '#edc02f',

        25: '#ff8444',
        26: '#a6dceb',
        27: '#fdd3a2',
        28: '#e6b1c2',
        29: '#9bb7d4',
        30: '#fedb5c',
        31: '#b2e1e0',
        32: '#f8c0b6',
        33: '#c8bfe7',
        34: '#f4af81',
        35: '#a3a3a3',
        36: '#bce784',
        37: '#8d6e63',
        38: '#e9e3c9',
        39: '#f5e9b2',
        40: '#ffba49',
        41: '#c0c0c0',
        42: '#d3a7b5',
        43: '#f2c2e0',
        44: '#b7dd29',
        45: '#dcf7c1',
        46: '#6f9ed7',
        47: '#d8a8c3',
        48: '#76c57f',
        49: '#f6e9cd',
        50: '#a16fd8',
        51: '#c5e6a7',
        52: '#f98f76',
        53: '#b3d6e3',
        54: '#efc8a5',
        55: '#5c9aa1',
        56: '#d3e1b6',
        57: '#a87ac8',
        58: '#e2d095',
        59: '#c95a3b',
        60: '#7fb4d1',
        61: '#f7d28e',
        62: '#b9c9b0',
        63: '#e994b9',
        64: '#8bc9e4',
        65: '#e6b48a',
        66: '#acd4d8',
        67: '#f3e0b0',
        68: '#57a773',
        69: '#d9bb7b',
        70: '#8e73e5',
        71: '#f4c4e3',
        72: '#75a88b',
        73: '#c0d4eb',
        74: '#a46c9b',
        75: '#d7e3a0',
        76: '#bd5f36',
        77: '#77c5b8',
        78: '#e8b7d5',
        79: '#4e8746',
        80: '#f0d695',
        81: '#9b75cc',
        82: '#c2e68a',
        83: '#f56e5c',
        84: '#a9ced0',
        85: '#e18a6d',
        86: '#6291b1',
        87: '#d1dbab',
        88: '#c376c5',
        89: '#8fc9b5',
        90: '#f7e39e',
        91: '#6d96b8',
        92: '#f9c0a6',
        93: '#63a77d',
        94: '#dbb8e9',
        95: '#9aa3d6',
        96: '#e3ca7f',
        97: '#b15d95',
        98: '#88c2e0',
        99: '#f4c995',
        100: '#507c94',
    }
    # color = {0: '#FF0000', 1: '#000000', 2: '#0000FF'}
    count = 0
    for data in gb_list:
        count = count+1
        if len(data)>1:
            # center and radius
            center_2d = data.mean(0)
            radius = np.max((((data - center_2d) ** 2).sum(axis=1) ** 0.5))
            center_3d = np.append(center_2d, 0)
            # data
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = radius * np.outer(np.cos(u), np.sin(v)) + center_3d[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + center_3d[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center_3d[2]
            # plot
            #fig = plt.figure()
            #ax = Axes3D(fig)
            ax.scatter(data[:, 0], data[:, 1], 0)
            
          
           # ax.plot_surface(x, y, z, rstride=4, cstride=4, alpha=0.2,color='b')
            ax.plot_wireframe(x, y, z,rstride=4, cstride=4, alpha=0.1,color=color[count])
        


# 遍历粒球是否需要分裂
def division(gb_list):
    gb_list_new = []
    for gb in gb_list:
        if len(gb) >4:
            density = get_density(gb)
            if density > 0.05:
            # if density > 0.0125:
                gb_list_new.extend(spilt_ball(gb))
            else:
                gb_list_new.append(gb)
        else:
            gb_list_new.append(gb)
    return gb_list_new




# 分裂粒球（kmeans）
def spilt_ball(data):
    # cluster = k_means(X=data, n_clusters=2, random_state=5)[1]
    cluster = k_means(X=data, n_clusters=2)[1]
    # print(cluster)
    ball1 = data[cluster == 0, :]
    ball2 = data[cluster == 1, :]
    return [ball1, ball2]


# 求距离
def distances(data, p):
    return ((data - p) ** 2).sum(axis=1) ** 0.5



def calculation_evenness(data, dot):
    dens1 = []
    dens2 = []
    for point in dot:
        dis_list = distances(data, point)
        dens1.append(min(dis_list))
        dens2.append(max(dis_list))
    evenness = (max(dens2) - min(dens1)) / max(dens2)
    return evenness

def get_density(gb):
    num = len(gb)
    center = gb.mean(0)
    diffMat = np.tile(center, (num, 1)) - gb
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    max_radius = max(distances)
    min_radius = min(distances)
    mean_radius = 0
    for i in distances:
        mean_radius = mean_radius + i
    mean_radius = mean_radius/num
    return (max_radius-mean_radius)


def get_radius(self):
    """
       Function function: calculate radius
   """
    diffMat = np.tile(self.center, (self.num, 1)) - self.data
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    return max(distances)

# 计算中心和半径和四个边界点
def calculation_center_and_radius(data):
    center = data.mean(0)
    radius = np.max((((data - center) ** 2).sum(axis=1) ** 0.5))
    # 找四个边界点的坐标
    dim = data.shape[1]

    coordinate = np.empty(shape=[0, dim])
    coordinate = np.vstack((coordinate, center))
    for i in range(dim):
        centdataitem = np.tile(center, (1, 1))
        centdataitem[:, i] = centdataitem[:, i] + radius
        coordinate = np.vstack((coordinate, centdataitem))

        centdataitem = np.tile(center, (1, 1))
        centdataitem[:, i] = centdataitem[:, i] - radius
        coordinate = np.vstack((coordinate, centdataitem))
    return coordinate.tolist()




# 绘制粒球
def gb_plot(gbs):
    color = {
        0: '#0000FF',
        1: '#FFFF00',
        2: '#FF0000',
        3: '#FFC0CB',
        4: '#8B0000',
        5: '#000000',
        6: '#008000',
        7: '#FFD700',
        8: '#A52A2A',
        9: '#FFA500',
        10: '#00FFFF',
        11: '#FF00FF',
        12: '#F0FFFF',
        13: '#00FFFF'}
    # 图像宽高与XY轴范围成比例，绘制粒球才是正圆
    plt.figure(figsize=(5, 5))  # 图像宽高
    plt.axis([-2, 2, -2, 2])  # 设置x轴的范围为[-1.2, 1.2]，y轴的范围为[-1, 1]
    # plt.axis([0, 1, 0, 1])
    label_num = {}
    for i in range(0, len(gbs)):
        label_num.setdefault(gbs[i].label,0)
        label_num[gbs[i].label] = label_num.get(gbs[i].label) + len(gbs[i].data)

    print("画图")

    label = set()
    for key in label_num.keys():
        if label_num[key]>10:
            label.add(key)
    list = []
    for i in range(0,len(label)):
        list.append(label.pop())
    for key in gbs.keys():
        # print(key)
        for i in range(0,len(list)):
            if(gbs[key].label == list[i]):
                if(i<14):
                    for data in gbs[key].data:
                        plt.plot(data[0], data[1],marker = '*',color= color[i], markersize=3)
                    break

    plt.show()

# 绘制粒球3d版本
def gb_3d_plot(gbs,ax):
    # color = {
    #     0: '#0000FF',
    #     1: '#FFFF00',
    #     2: '#FF0000',
    #     3: '#FFC0CB',
    #     4: '#8B0000',
    #     5: '#000000',
    #     6: '#008000',
    #     7: '#FFD700',
    #     8: '#A52A2A',
    #     9: '#FFA500',
    #     10: '#00FFFF',
    #     11: '#FF00FF',
    #     12: '#F0FFFF',
    #     13: '#00FFFF'}
    # 图像宽高与XY轴范围成比例，绘制粒球才是正圆
    color = {
        0: '#707afa',
        1: '#ffe135',
        2: '#16ccd0',
        3: '#ed7231',
        4: '#0081cf',
        5: '#afbed1',
        6: '#bc0227',
        7: '#d4e7bd',
        8: '#f8d7aa',
        9: '#fecf45',
        10: '#f1f1b8',
        11: '#b8f1ed',
        12: '#ef5767',
        13: '#e7bdca',
        14: '#8e7dfa',
        15: '#d9d9fc',
        16: '#2cfa41',
        17: '#e96d29',
        18: '#7f722f',
        19: '#bd57fa',
        20: '#e4f788',
        21: '#fb8e94',
        22: '#b8d38f',
        23: '#e3a04f',
        24: '#edc02f',

        25: '#ff8444',
        26: '#a6dceb',
        27: '#fdd3a2',
        28: '#e6b1c2',
        29: '#9bb7d4',
        30: '#fedb5c',
        31: '#b2e1e0',
        32: '#f8c0b6',
        33: '#c8bfe7',
        34: '#f4af81',
        35: '#a3a3a3',
        36: '#bce784',
        37: '#8d6e63',
        38: '#e9e3c9',
        39: '#f5e9b2',
        40: '#ffba49',
        41: '#c0c0c0',
        42: '#d3a7b5',
        43: '#f2c2e0',
        44: '#b7dd29',
        45: '#dcf7c1',
        46: '#6f9ed7',
        47: '#d8a8c3',
        48: '#76c57f',
        49: '#f6e9cd',
        50: '#a16fd8',
        51: '#c5e6a7',
        52: '#f98f76',
        53: '#b3d6e3',
        54: '#efc8a5',
        55: '#5c9aa1',
        56: '#d3e1b6',
        57: '#a87ac8',
        58: '#e2d095',
        59: '#c95a3b',
        60: '#7fb4d1',
        61: '#f7d28e',
        62: '#b9c9b0',
        63: '#e994b9',
        64: '#8bc9e4',
        65: '#e6b48a',
        66: '#acd4d8',
        67: '#f3e0b0',
        68: '#57a773',
        69: '#d9bb7b',
        70: '#8e73e5',
        71: '#f4c4e3',
        72: '#75a88b',
        73: '#c0d4eb',
        74: '#a46c9b',
        75: '#d7e3a0',
        76: '#bd5f36',
        77: '#77c5b8',
        78: '#e8b7d5',
        79: '#4e8746',
        80: '#f0d695',
        81: '#9b75cc',
        82: '#c2e68a',
        83: '#f56e5c',
        84: '#a9ced0',
        85: '#e18a6d',
        86: '#6291b1',
        87: '#d1dbab',
        88: '#c376c5',
        89: '#8fc9b5',
        90: '#f7e39e',
        91: '#6d96b8',
        92: '#f9c0a6',
        93: '#63a77d',
        94: '#dbb8e9',
        95: '#9aa3d6',
        96: '#e3ca7f',
        97: '#b15d95',
        98: '#88c2e0',
        99: '#f4c995',
        100: '#507c94',
    }
    plt.figure(figsize=(5, 5))  # 图像宽高
    plt.axis([-2, 2, -2, 2])  # 设置x轴的范围为[-1.2, 1.2]，y轴的范围为[-1, 1]
    # plt.axis([0, 1, 0, 1])
    label_num = {}
    for i in range(0, len(gbs)):
        label_num.setdefault(gbs[i].label,0)
        label_num[gbs[i].label] = label_num.get(gbs[i].label) + len(gbs[i].data)

    print("画图")

    label = set()
    for key in label_num.keys():
        if label_num[key]>10:
            label.add(key)
    list = []
    for i in range(0,len(label)):
        list.append(label.pop())
    for key in gbs.keys():
        # print(key)
        for i in range(0,len(list)):
            if(gbs[key].label == list[i]):
                if(i<14):
                    for data in gbs[key].data:
                        #plt.plot(data[0], data[1],marker = '*',color= color[i], markersize=3)
                        ax.scatter(data[0], data[1],color=color[i])
                    break

    plt.show()


def add_center(gb_list,ax):
    gb_dist = {}
    for i in range(0,len(gb_list)):
        gb = GB(gb_list[i],i)
        gb_dist[i] = gb


    radius_sum = 0
    num_sum = 0
    


    gblen = 0
    radius_sum = 0
    num_sum = 0
    for i in range(0, len(gb_dist)):
        if gb_dist[i].out == 0:
            gblen = gblen + 1
            radius_sum = radius_sum + gb_dist[i].radius
            num_sum = num_sum + gb_dist[i].num



    gb_uf = UF(len(gb_list))
    for i in range(0, len(gb_dist)-1):
        if gb_dist[i].out != 1:
            center = gb_dist[i].center
            radius = gb_dist[i].radius
            for j in range(i + 1, len(gb_dist)):
                if gb_dist[j].out != 1:
                    center2 = gb_dist[j].center
                    radius2 = gb_dist[j].radius
                    max_radius = max(radius, radius2)
                    min_radius = min(radius, radius2)
                    dis = ((center - center2) ** 2).sum(axis=0) ** 0.5
                    if dis <= radius + radius2 + radius_sum/(gblen):
                    # if dis <= (radius + radius2) + (min_radius+max_radius)/2:
                        gb_dist[i].flag = 1
                        gb_dist[j].flag = 1
                        gb_uf.union(i,j)

    for i in range(0, len(gb_dist)):
        k = i
        if gb_uf.parent[i] != i:
            while(gb_uf.parent[k]!=k):
                k = gb_uf.parent[k]
        gb_uf.parent[i] = k





    for i in range(0, len(gb_dist)):
        gb_dist[i].label = gb_uf.parent[i]
        gb_dist[i].size = gb_uf.size[i]


    print("哈哈哈")
    label_num = set()
    for i in range(0, len(gb_dist)):
        label_num.add(gb_dist[i].label)
    print(len(label_num))
    # gb_plot(gb_dist)



    for i in range(0, len(gb_dist)):
        distance = 1
        if gb_dist[i].flag == 0:
            for j in range(0, len(gb_dist)):
                if gb_dist[j].flag == 1:
                    center = gb_dist[i].center
                    center2 = gb_dist[j].center
                    radius = gb_dist[i].radius
                    radius2 = gb_dist[j].radius
                    dis = ((center - center2) ** 2).sum(axis=0) ** 0.5
                    if dis < distance:
                        distance = dis
                        gb_dist[i].label = gb_dist[j].label



    label_num = set()
    for i in range(0, len(gb_dist)):
        label_num.add(gb_dist[i].label)
    print(len(label_num))
    print("哈哈哈")
    gb_3d_plot(gb_dist,ax)



def main(evenness = 0.5):

    #keys = ['core vs nocore','cth','DS4-2','DS6-2','fivecluster','ls','spiral','spiralunbalance','ThreeCircles','threecluster','twocluster','Twomoons']
    
    keys = ['1','2','3','4','5','6','7','8','9','10','11','12']
    keys = ['4']

    for d in range(len(keys)):
        print(keys[d])
        df = pd.read_csv(r"../dataset/" + keys[d] + ".csv", header=None)  # 加载数据集
        data = df.values
        # data = data[:,1:]
        data = np.unique(data, axis=0)
        scaler = StandardScaler()
        # scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        # fig = plt.figure()
        # ax = Axes3D(fig)
        gb_list = [data]
        # draw_3d_ball(gb_list,ax)
        # plt.axis('off')
        # plt.show()
        count = 0
        while 1:
            ball_number_old = len(gb_list)
            gb_list = division(gb_list)
            ball_number_new = len(gb_list)
            count = count + 1
            # fig = plt.figure()
            # ax = Axes3D(fig)
            # if ball_number_new == 2 or ball_number_new == 4:
            # draw_3d_ball(gb_list, ax)
            # plt.axis('off')
            # plt.show()
            if ball_number_new == ball_number_old:
                break
        print("粒球数量", ball_number_new)
        fig = plt.figure()

        ax = Axes3D(fig)
        draw_3d_ball(gb_list, ax)
        plt.axis('off')
        plt.show()
        add_center(gb_list,ax)

        print("sucess")



if __name__ == '__main__':
    main()