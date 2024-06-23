from gbutils.granular_ball import GranularBall
from gbutils.HyperballClustering import *


def splitGBs(data):
    gb_list_temp = [data]  # 粒球集合[ [[data1],[data2],...], [[data1],[data2],...],... ],初始只有一个粒球
    division_num = 0  # 记录第几次分裂
    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = division_2_2(gb_list_temp)  # 粒球划分
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break

    radius = []  # 汇总所有粒球半径
    for gb_data in gb_list_temp:
        if len(gb_data) >= 2:  # 粒球中样本多于2个时，认定为合法粒球，并收集其粒球半径
            radius.append(get_radius(gb_data[:, :]))
    radius_median = np.median(radius)
    radius_mean = np.mean(radius)
    radius_detect = min(radius_median, radius_mean)

    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = minimum_ball(gb_list_temp, radius_detect)  # 缩小粒球
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break
    gb_list = []
    for obj in gb_list_temp:
        # 去噪声，移除1个点的球
        if len(obj) == 1:
            continue
        gb = GranularBall(obj)
        gb_list.append(gb)
    return gb_list


