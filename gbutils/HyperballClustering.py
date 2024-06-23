# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:35:51 2022

@author: xiejiang
"""
from matplotlib.ticker import FixedLocator
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
matplotlib.use('TkAgg')  # 或者 'Agg', 'Qt5Agg', 'Qt4Agg', 'WXAgg', 'GTK3Agg'


class GB:
    def __init__(self, data, label):
        self.data = data
        self.center = self.data[:, :-1].mean(0)
        self.radius = self.get_radius()
        self.label = label
        self.num = len(data)

    def get_radius(self):
        return max(((self.data[:, :-1] - self.center) ** 2).sum(axis=1) ** 0.5)


# 粒球划分
def division(hb_list, n):
    gb_list_new = []
    for hb in hb_list:
        if len(hb) >= 8:
            ball_1, ball_2 = spilt_ball(hb)  # 粒球分裂
            DM_parent = get_DM(hb)
            DM_child_1 = get_DM(ball_1)
            DM_child_2 = get_DM(ball_2)
            t1 = ((DM_child_1 > DM_parent) & (DM_child_2 > DM_parent))
            if t1:
                gb_list_new.extend([ball_1, ball_2])
            else:
                gb_list_new.append(hb)
        else:
            gb_list_new.append(hb)

    return gb_list_new


def spilt_ball_2(data):
    ball1 = []
    ball2 = []
    n, m = data.shape
    X = data.T
    G = np.dot(X.T, X)
    H = np.tile(np.diag(G), (n, 1))
    D = np.sqrt(np.abs(H + H.T - G * 2))
    r, c = np.where(D == np.max(D))
    r1 = r[1]
    c1 = c[1]
    for j in range(0, len(data)):
        if D[j, r1] < D[j, c1]:
            ball1.extend([data[j, :]])
        else:
            ball2.extend([data[j, :]])
    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2]


def get_density_volume(gb):
    num = len(gb)

    center = gb.mean(0)
    diffMat = np.tile(center, (num, 1)) - gb
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sum_radius = 0
    if len(distances) == 0:
        print("0")
    # radius = max(distances)
    for i in distances:
        sum_radius = sum_radius + i
    mean_radius = sum_radius / num
    dimension = len(gb[0])
    # print('*******dimension********',dimension)
    if mean_radius != 0:
        # density_volume = num/(radius**dimension)
        # density_volume = num/((radius**dimension)*sum_radius)
        density_volume = num / sum_radius
        # density_volume = num/(sum_radius)
    else:
        density_volume = num

    return density_volume


# 无参遍历粒球是否需要分裂，根据子球和父球的比较，不带断裂判断的分裂,1分2
def division_2_2(gb_list):
    gb_list_new = []
    for gb_data in gb_list:
        if len(gb_data) >= 8:
            ball_1, ball_2 = spilt_ball_2(gb_data)  # 无模糊
            # ball_1, ball_2 = spilt_ball_fuzzy(gb_data)  # 有模糊
            # # ?
            # if len(ball_1) * len(ball_2) == 0:
            #     return gb_list
            # # ?
            # (zt)6.9
            if len(ball_1) == 1 or len(ball_2) == 1:
                gb_list_new.append(gb_data)
                continue
            if len(ball_1) == 0 or len(ball_2) == 0:
                gb_list_new.append(gb_data)
                continue
            # if len(ball_1)*len(ball_2) == 0:
            #     gb_list_new.append(gb_data)
            #     continue
            # print("p")
            # print(len(gb_data[:, :-1]))
            # print("b1")
            # print(len(ball_1[:, :-1]))
            # print("b2")
            # print(len(ball_2[:, :-1]))
            parent_dm = get_density_volume(gb_data[:, :])
            child_1_dm = get_density_volume(ball_1[:, :])
            child_2_dm = get_density_volume(ball_2[:, :])
            w1 = len(ball_1) / (len(ball_1) + len(ball_2))
            w2 = len(ball_2) / (len(ball_1) + len(ball_2))
            # sep = get_separation(ball_1,ball_2)
            # print('this is sep test',sep)
            # t = (w1*density_child_1+w2*density_child_2)- 1/(w1*w2)*(w/n)
            w_child_dm = (w1 * child_1_dm + w2 * child_2_dm)  # 加权子粒球DM
            # if w > 20:
            # print("_______________________")
            # print("父亲数量", len(gb_data))
            # print("子球1数量", len(ball_1))
            # print("子球2数量", len(ball_2))
            # print("父球质量", parent_dm)
            # print("子球1质量", child_1_dm)
            # print("子球2质量", child_2_dm)
            # print("子球加权质量", w_child_dm)
            t1 = ((child_1_dm > parent_dm) & (child_2_dm > parent_dm))
            t2 = (w_child_dm > parent_dm)  # 加权DM上升
            t3 = ((len(ball_1) > 0) & (len(ball_2) > 0))  # 球中数据个数低于4个的情况不能分裂
            if t2:
                gb_list_new.extend([ball_1, ball_2])
            else:
                # print("父亲数量",w)
                # print("子球1数量",len(ball_1))
                # print("子球2数量",len(ball_2))
                # print("父球质量",density_parent)
                # print("子球1质量",density_child_1)
                # print("子球2质量",density_child_2)
                # print("分裂后效果不好")
                gb_list_new.append(gb_data)
        else:
            gb_list_new.append(gb_data)

    return gb_list_new


# 粒球分裂
def spilt_ball(data):
    ball1 = []
    ball2 = []
    n, m = data.shape
    X = data.T
    G = np.dot(X.T, X)
    H = np.tile(np.diag(G), (n, 1))
    D = np.sqrt(np.abs(H + H.T - G * 2))  # 计算欧式距离
    r, c = np.where(D == np.max(D))  # 查找距离最远的两个点坐标
    r1 = r[1]
    c1 = c[1]
    for j in range(0, len(data)):  # 对于距离最远的两个点之外的点，根据到两点距离不同划分到不同的簇中
        if D[j, r1] < D[j, c1]:
            ball1.extend([data[j, :]])
        else:
            ball2.extend([data[j, :]])
    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2]



def get_DM(hb):
    num = len(hb)
    center = hb.mean(0)
    diffMat = np.tile(center, (num, 1)) - hb
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5  # 欧式距离
    sum_radius = 0
    radius = max(distances)
    for i in distances:
        sum_radius = sum_radius + i
    mean_radius = sum_radius / num
    dimension = len(hb[0])
    if mean_radius != 0:
        DM = num / sum_radius
    else:
        DM = num
    return DM


def get_radius(gb_data):
    sample_num = len(gb_data)
    center = gb_data.mean(0)
    diffMat = np.tile(center, (sample_num, 1)) - gb_data
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    radius = max(distances)
    return radius



def ball_tree(samples):
    # 1. 计算中心点 o
    o = np.mean(samples, axis=0)

    # 2. 找到离 o 最远的样本 x1
    distances_from_o = np.linalg.norm(samples - o, axis=1)
    x1 = samples[np.argmax(distances_from_o)]

    # 3. 找到离 x1 最远的样本 x2
    distances_from_x1 = np.linalg.norm(samples - x1, axis=1)
    x2 = samples[np.argmax(distances_from_x1)]

    # 4. 计算中点 o1 和 o2
    o1 = (x1 + o) / 2
    o2 = (x2 + o) / 2
    return [o, x1, x2, o1, o2]


# 缩小粒球
def minimum_ball(gb_list, radius_detect):
    gb_list_temp = []
    for gb_data in gb_list:
        # if len(hb) < 2: stream
        if len(gb_data) <= 2:
            # gb_list_temp.append(gb_data)

            if (len(gb_data) == 2) and (get_radius(gb_data) > 1.2 * radius_detect):
                # print(get_radius(gb_data))
                gb_list_temp.append(np.array([gb_data[0], ]))
                gb_list_temp.append(np.array([gb_data[1], ]))

            else:
                gb_list_temp.append(gb_data)
        else:
            # if get_radius(gb_data) <= radius_detect:
            if get_radius(gb_data) <= 1.2 * radius_detect:
                gb_list_temp.append(gb_data)
            else:
                ball_1, ball_2 = spilt_ball(gb_data)  # 无模糊
                # ball_1, ball_2 = spilt_ball_fuzzy(gb_data)  # 有模糊
                if len(ball_1) == 1 or len(ball_2) == 1:
                    if get_radius(gb_data) > radius_detect:
                        gb_list_temp.extend([ball_1, ball_2])
                    else:
                        gb_list_temp.append(gb_data)
                else:
                    gb_list_temp.extend([ball_1, ball_2])

    return gb_list_temp



