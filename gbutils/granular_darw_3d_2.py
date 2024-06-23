# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 14:15:32 2021

@author: xiejiang
"""
# best

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
        # self.init_center = self.random_center()  # Get a random point in each tag
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


def get_radius_2(gb):
    num = len(gb)
    center = gb.mean(0)
    diffMat = np.tile(center, (num, 1)) - gb
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    mean_radius = 0
    radius = max(distances)
    for i in distances:
        mean_radius = mean_radius + i
    mean_radius = mean_radius/num
    return radius
    
# 判断两个粒球是都断裂   
def judge_break(ball_1,ball_2):
    radius_1 = get_radius_2(ball_1)
    radius_2 = get_radius_2(ball_2)
    center_1 = ball_1.mean(0)
    center_2 = ball_2.mean(0)
    distance_center = np.linalg.norm(center_1-center_2)
    distance_radius = radius_1+radius_2
    if distance_center > distance_radius:
        return 0
    else:
        return 1



# 遍历粒球是否需要分裂
def division(gb_list):
    gb_list_new = []
    for gb in gb_list:
        if len(gb) >4:
            density = get_density(gb)
            if density > 0.025:
            # if density > 0.0125:
                gb_list_new.extend(spilt_ball(gb))
            else:
                gb_list_new.append(gb)
        else:
            gb_list_new.append(gb)
    return gb_list_new


# 无参遍历粒球是否需要分裂，根据子球和父球的比较，不带断裂判断的分裂
def division_2(gb_list,n):
    
    gb_list_new = []
    
    for gb in gb_list:
        if ((len(gb) > 3) & (len(gb_list) <= (n**0.5))):#因为分裂为三个球，为了分裂的每个球有4个以上的数据，总共有12个
        #if ((len(gb_list) <= (n**0.5))):
        #if (len(gb) > 3):
            print('********************')
            ball_1,ball_2,ball_3= spilt_ball(gb)
            density_parent = get_density_volume(gb)
            density_child_1 = get_density_volume(ball_1)
            density_child_2 = get_density_volume(ball_2)
            density_child_3 = get_density_volume(ball_3)
            w = len(ball_1)+len(ball_2)+len(ball_3)
            w1 = len(ball_1)/w
            w2 = len(ball_2)/w
            w3 = len(ball_3)/w
            print("父球质量",density_parent)
            print("子球1质量",density_child_1)
            print("子球2质量",density_child_2)
            print("子球3质量",density_child_3)
            print("子球加权质量",w1*density_child_1+w2*density_child_2+w3*density_child_3)
            print(len(gb))
            print(len(ball_1))
            print(len(ball_2))
            print(len(ball_3))
            print('********************')
            t1 = density_parent
            t2 = (w1*density_child_1+w2*density_child_2+w3*density_child_3)
            t3 =  (t2 > t1)#父球和子球的质量比较条件
            c1 = ((len(ball_1) > 4) & (len(ball_2) > 4) & (len(ball_3) > 4))# 数据个数条件
            if (t3):
                gb_list_new.extend([ball_1,ball_2,ball_3])
            else:
                gb_list_new.append(gb)
        else:
            gb_list_new.append(gb)
        
    return gb_list_new

#在分裂到根号n个粒球后，如果分裂后的子球不断裂则允许继续分裂细化，每次分裂为3个
def division_3_1(gb_list,n):
    
    gb_list_new_2 = []
    gb_list_new = division_2(gb_list,n)
    
    for gb in gb_list_new:
        if (len(gb) > 4): #因为分裂为三个球，为了分裂的每个球有4个以上的数据，总共有12个
           ball_1,ball_2,ball_3= spilt_ball(gb)
           density_parent = get_density_volume(gb)
           density_child_1 = get_density_volume(ball_1)
           density_child_2 = get_density_volume(ball_2)
           density_child_3 = get_density_volume(ball_3)
           break_1 = judge_break(ball_1,ball_2)
           break_2 = judge_break(ball_1,ball_3)
           break_3 = judge_break(ball_2,ball_3)
           w = len(ball_1)+len(ball_2)+len(ball_3)
           w1 = len(ball_1)/w
           w2 = len(ball_2)/w
           w3 = len(ball_3)/w
           t2 = (w1*density_child_1+w2*density_child_2+w3*density_child_3)
           t1 = density_parent
           t3 = (t2 > t1) #子球与父球质量对比
           b1 = (break_1 & break_2) # 分裂后的3个子球 只要能两两重叠就满足条件
           b2 = (break_1 & break_3)
           b3 = (break_2 & break_3)
           c1 = ((len(ball_1) > 4) & (len(ball_2) > 4) & (len(ball_3) > 4))
           # if (break_1 & break_2 & break_3)&(density_parent < t):
           # if (break_1 & break_2 & break_3):
           if (b1 | b2 | b3) :
               gb_list_new_2.extend([ball_1,ball_2,ball_3])
           else:
               gb_list_new_2.append(gb)
        else:
          gb_list_new_2.append(gb)  
        
    return gb_list_new_2


# 无参遍历粒球是否需要分裂，根据子球和父球的比较，不带断裂判断的分裂
def division_2_1(gb_list,n):
    
    gb_list_new = []
    
    for gb in gb_list:
        # if ((len(gb) > 3) & (len(gb_list) <= ((n**0.5)))):#因为分裂为三个球，为了分裂的每个球有4个以上的数据，总共有12个
        #if ((len(gb_list) <= (n**0.5))):
        if (len(gb) > 3):
            # print('********************')
            ball_1,ball_2,ball_3= spilt_ball(gb)
            density_parent = get_density_volume(gb)
            density_child_1 = get_density_volume(ball_1)
            density_child_2 = get_density_volume(ball_2)
            density_child_3 = get_density_volume(ball_3)
            w = len(ball_1)+len(ball_2)+len(ball_3)
            w1 = len(ball_1)/w
            w2 = len(ball_2)/w
            w3 = len(ball_3)/w
            # print("父球质量",density_parent)
            # print("子球1质量",density_child_1)
            # print("子球2质量",density_child_2)
            # print("子球3质量",density_child_3)
            # print("子球加权质量",w1*density_child_1+w2*density_child_2+w3*density_child_3)
            # print(len(gb))
            # print(len(ball_1))
            # print(len(ball_2))
            # print(len(ball_3))
            # print('********************')
            t1 = density_parent
            t2 = (w1*density_child_1+w2*density_child_2+w3*density_child_3)
            t3 =  (t2 >= t1)#父球和子球的质量比较条件
            t4 = ((len(ball_1)>2)&(len(ball_2)>2)&(len(ball_3)>2))# 球中数据个数只有2个的情况不能分裂
            # print("len(ball1),len(ball2)，len(ball3),t4",len(ball_1),len(ball_2),len(ball_3),t4)
            if (t3 & t4):
                # print("执行分裂")
                gb_list_new.extend([ball_1,ball_2,ball_3])
            else:
                gb_list_new.append(gb)
                # print("不分裂")
        else:
            gb_list_new.append(gb)
        
    return gb_list_new

def division_2_2(gb_list,n):
    
    gb_list_new_2 = []

    
    for gb in gb_list:
        if(len(gb)>4):
        # if ((len(gb) > 3)&(len(gb_list) <= ((n**0.5)))):
           ball_1,ball_2 = spilt_ball_2(gb)
           density_parent = get_density_volume(gb)
           density_child_1 = get_density_volume(ball_1)
           density_child_2 = get_density_volume(ball_2)
           w = len(ball_1)+len(ball_2)
           w1 = len(ball_1)/w
           w2 = len(ball_2)/w
           t = w1*density_child_1+w2*density_child_2
           if w>130:
               print("父亲数量",w)
               print("子球1数量",len(ball_1))
               print("子球2数量",len(ball_2))
               print("父球质量",density_parent)
               print("子球1质量",density_child_1)
               print("子球2质量",density_child_2)
               print("子球加权质量",t)
           t4 = ((len(ball_1) >=4) & (len(ball_2) >=4)) # 球中数据个数只有一个的情况不能分裂
           if (density_parent < t)&(t4):
               gb_list_new_2.extend([ball_1,ball_2])
           else:
               gb_list_new_2.append(gb)
        else:
          gb_list_new_2.append(gb)  
        
    return gb_list_new_2


#在分裂到根号n个粒球后，如果分裂后的子球不断裂则允许继续分裂细化，每次分裂为2个
def division_3_2(gb_list,n):
    
    gb_list_new_2 = []
    gb_list_new = division_2(gb_list,n)
    
    for gb in gb_list_new:
        if (len(gb) > 4):
           ball_1,ball_2 = spilt_ball_2(gb)
           density_parent = get_density_volume(gb)
           density_child_1 = get_density_volume(ball_1)
           density_child_2 = get_density_volume(ball_2)
           break_1 = judge_break(ball_1,ball_2)
           w = len(ball_1)+len(ball_2)
           w1 = len(ball_1)/w
           w2 = len(ball_2)/w
           t = w1*density_child_1+w2*density_child_2
           #if (break_1) & (density_parent < t):
           if (break_1):
               gb_list_new_2.extend([ball_1,ball_2])
           else:
               gb_list_new_2.append(gb)
        else:
          gb_list_new_2.append(gb)  
        
    return gb_list_new_2


# 判断粒球是否存在包含关系

def judge_contain(gb_target,gb_list):
    gb_list_new = []
    center_target = gb_target.mean(0)
    radius_target = get_radius(gb_target)
    for gb in gb_list:
        if(gb is gb_target):
            continue
        else:
            center_gb = gb.mean(0)
            radius_gb = get_radius(gb)
            distance_1 =  np.linalg.norm(center_target-center_gb)
            if(radius_target >= (radius_gb + distance_1)):
                return 1
    return 0



# 去掉有包含关系的球        
def move_contain(gb_list):
    gb_list_final=[]
    for gb in gb_list:
        is_contain = judge_contain(gb,gb_list)
        if (is_contain):
            gb_list_final.extend(spilt_ball(gb))
            print("有包含关系",len(gb_list_final))
        else:
            gb_list_final.append(gb)
            
    return gb_list_final



    
# 分裂粒球（kmeans）分裂默认为3个
def spilt_ball(data):
    # cluster = k_means(X=data, n_clusters=2, random_state=5)[1]
    cluster = k_means(X=data, init='k-means++',n_clusters=3)[1]  # 中心选择最远的点
    # print(cluster)
    ball1 = data[cluster == 0, :]
    ball2 = data[cluster == 1, :]
    ball3 = data[cluster == 2, :]
    return [ball1, ball2,ball3]



# 分裂粒球（kmeans）分裂为2个
def spilt_ball_2(data):
    # cluster = k_means(X=data, n_clusters=2, random_state=5)[1]
    cluster = k_means(X=data, n_clusters=2)[1]
    # print(cluster)
    ball1 = data[cluster == 0, :]
    ball2 = data[cluster == 1, :]
    return [ball1, ball2]



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

def get_density_volume(gb):
    num = len(gb)
    center = gb.mean(0)
    diffMat = np.tile(center, (num, 1)) - gb
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    mean_radius = 0
    radius = max(distances)
    for i in distances:
        mean_radius = mean_radius + i
    mean_radius = mean_radius/num
    dimension = len(gb[0])
    density_volume = num/(mean_radius**dimension)
    print(mean_radius**dimension)
    return density_volume
 

    

# 获取一个粒球的半径
def get_radius(gb):
    num = len(gb)
    center = gb.mean(0)
    diffMat = np.tile(center, (num, 1)) - gb
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    radius = max(distances)
    return radius
    
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

# def overlap(gb_list):
#     print(gb_list[0])
#     print("suuc")

def plot_dot(data):
    color = {0: '#FF0000', 1: '#000000', 2: '#0000FF'}
    plt.figure(figsize=(6, 6))
    # xMin = min(data[:, 0])
    # xMax = max(data[:, 0])
    # yMin = min(data[:, 1])
    # yMax = max(data[:, 1])
    # plt.axis([xMin, xMax, yMin, yMax])
    plt.axis([-2, 2, -2, 2])
    # plt.axis([0, 1, 0, 1])
    plt.plot(data[:, 0], data[:, 1], '.', color=color[0], markersize=3)

#绘制数据点3d图
def plot_3d_dot(data,ax):
 
    ax.scatter(data[:, 0], data[:, 1],0)


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
                        #plt.plot(data[0], data[1],marker = '*',color= color[i], markersize=3)
                        ax.axis('off')
                        ax.scatter(data[0], data[1],color=color[i])
                    break
     
    
    plt.show()



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
            plt.plot(data[0][0], data[0][1],marker = '*',color= '#0000EF', markersize=3)
    plt.show()
    

#绘制3d粒球
def draw_3d_ball(gb_list,ax):
    color = {0: '#FF0000', 1: '#000000', 2: '#0000FF'}
    count = 0
    for data in gb_list:
        count = count+1
        if len(data)>1:
            # center and radius
            center_2d = data.mean(0)
            radius = np.max((((data - center_2d) ** 2).sum(axis=1) ** 0.5))
            center_3d = np.append(center_2d,0)
            # data
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = radius * np.outer(np.cos(u), np.sin(v)) + center_3d[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + center_3d[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center_3d[2]
            # plot
            #fig = plt.figure()
            #ax = Axes3D(fig)
            ax.scatter(data[:, 0], data[:, 1],0)
            
          
           # ax.plot_surface(x, y, z, rstride=4, cstride=4, alpha=0.2,color='b')
            ax.plot_wireframe(x, y, z,rstride=4, cstride=4, alpha=0.1,color=color[count%3])
            
            
# 绘制3d结果
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
    """
        Function function: according to a certain range of purity threshold, get the particle partition under each purity threshold
        Input: training set sample, training set label, minimum purity threshold, maximum purity threshold
        Output: sample after sampling within each purity threshold range, sample label after sampling within each purity threshold range
    """


   # keys = ['1','2','3','4','5','6','7','8','9','10','11','12']
   # keys = ['a1','a2','a3','a4','a5']
    keys=['4']

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
        
        gb_list_1 = [data]
        
        # draw_3d_ball(gb_list_1,ax)
        # plt.axis('off')
        # plt.show()
        
        row = np.shape(gb_list_1)[0]
        col = np.shape(gb_list_1)[1]
        n = row*col
    
        # b1,b2,b3 = spilt_ball(gb_list[0]) #针对对称的数据集先进行一次分裂
        # gb_list_1=[]
        # gb_list_1.extend([b1,b2,b3])
        
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # draw_3d_ball(gb_list_1,ax)
        # plt.axis('off')
        # plt.show()
        
        while 1:
            ball_number_old = len(gb_list_1)
            gb_list_1 = division_2_2(gb_list_1,n)
            ball_number_new = len(gb_list_1)
            
            # fig = plt.figure()
            # ax = Axes3D(fig)
            # if ball_number_new == 2 or ball_number_new == 4:
            #     draw_3d_ball(gb_list_1,ax)
            # plt.axis('off')
            # plt.show()
            
            print("ball_number_old:",ball_number_old)
            print("ball_number_new:",ball_number_new)
            if ball_number_new == ball_number_old:
                break
            
            
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # gb_list_1 = move_contain(gb_list_1)
        # print("ball_number_final",len(gb_list_1))
        

        # draw_3d_ball(gb_list_1,ax)
        # plt.axis('off')
        # plt.show()
       
        fig = plt.figure()
        ax = Axes3D(fig)
        print("粒球数量", len(gb_list_1))
        draw_3d_ball(gb_list_1, ax)
        plt.axis('off')
        add_center(gb_list_1,ax)
       
        print("clusterng complete")
       
    
        



if __name__ == '__main__':
    main()