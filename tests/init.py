# coding=utf-8
import pickle, sys

sys.path.append("../")

# from simulator.utilities import *
from alg_utility import *
from envs import *


def find(mapped_matrix_int, a):
    M, N = mapped_matrix_int.shape
    for i in range(M):
        for j in range(N):
            if a == mapped_matrix_int[i][j]: return i, j


# 加入随机生成代码
import networkx as nx
import numpy as np
import random

G = nx.Graph()
distance = []
crime = []
for i in range(26):
    G.add_node(i)
for i in range(26):
    D = []
    C = []
    for j in range(26):
        if (j < i):
            D.append(distance[j][i])
            C.append(crime[j][i])
        else:
            if (i != j):
                flag = random.randint(1, 100)
                if flag < 35:
                    d = random.randint(1, 9)
                    c = random.randint(0, 5)
                    D.append(d)
                    C.append(c)
                    G.add_edge(i, j, weight=d + c * 50)
                else:
                    D.append(0)
                    C.append(0)
            else:
                D.append(0)
                C.append(0)
    distance.append(D)
    crime.append(C)
distance = np.array(distance)
crime = np.array(crime)

print "distance{}".format(distance)

print "crime{}".format(crime)

################## Load data ###################################
dir_prefix = ""
current_time = time.strftime("%Y%m%d_%H-%M")
log_dir = dir_prefix + "dispatch_simulator/experiments/{}/".format(current_time)
mkdir_p(log_dir)
print "log dir is {}".format(log_dir)

data_dir = dir_prefix + "dispatch_realdata/data_for_simulator/"
order_time_dist = []  # 原来是空的
order_price_dist = []
mapped_matrix_int = np.array(
    [[5, 8, 2, 17, 15], [1, 7, 3, 23, 19], [6, 4, 9, 11, 22], [16, 10, 13, -100, 21], [12, 14, 18, 20, 24]])
# 点和原来不一样
M, N = mapped_matrix_int.shape
order_num_dist = []
num_valid_grid = 24
idle_driver_location_mat = np.zeros((144, 24))

for ii in np.arange(144):
    time_dict = {}
    for jj in np.arange(M * N):  # num of grids
        time_dict[jj] = [2]
    order_num_dist.append(time_dict)
    idle_driver_location_mat[ii, :] = [2] * num_valid_grid

idle_driver_dist_time = [[10, 1] for _ in np.arange(144)]

n_side = 6
l_max = 2
order_time = [0.2, 0.2, 0.15,
              0.15, 0.1, 0.1,
              0.05, 0.04, 0.01]
order_price = [[10.17, 3.34],  # mean and std of order price when duration is 10 min
               [15.02, 6.90],  # mean and std of order price when duration is 20 min
               [23.22, 11.63],
               [32.14, 16.20],
               [40.99, 20.69],
               [49.94, 25.61],
               [58.98, 31.69],
               [68.80, 37.25],
               [79.40, 44.39]]

order_real = []
onoff_driver_location_mat = []
for tt in np.arange(144):
    for i in range(1, 25):
        a = []
        for j in range(1, 25):
            flag = random.randint(1, 1000)
            if flag < 50:
                if distance[i][j] != 0:
                    [row1, col1] = find(mapped_matrix_int, i)
                    [row2, col2] = find(mapped_matrix_int, j)
                    b = row1 - row2
                    if b < 0: b = -b
                    c = col1 - col2
                    if c < 0: c = -c
                    a.append([i, j, tt, 10, 100 - distance[i][j] - crime[i][j] * 10])
        order_real += a

    onoff_driver_location_mat.append([[-0.625, 2.92350389],
                                      [0.09090909, 1.46398452],
                                      [0.09090909, 2.36596622],
                                      [0.09090909, 2.36596622],
                                      [0.09090909, 1.46398452],
                                      [0.09090909, 1.46398452],
                                      [0.09090909, 1.46398452],
                                      [0.09090909, 1.46398452],
                                      [0.09090909, 1.46398452],
                                      [0.09090909, 2.36596622],
                                      [0.09090909, 2.36596622],
                                      [0.09090909, 1.46398452],
                                      [0.09090909, 1.46398452],
                                      [0.09090909, 1.46398452],
                                      [0.09090909, 1.46398452],
                                      [0.09090909, 1.46398452],
                                      [0.09090909, 2.36596622],
                                      [0.09090909, 2.36596622],
                                      [0.09090909, 1.46398452],
                                      [0.09090909, 1.46398452],
                                      [0.09090909, 1.46398452],
                                      [0.09090909, 1.46398452],
                                      [0.09090909, 1.46398452],
                                      [0.09090909, 2.36596622],
                                      [0.09090909, 1.46398452]])

print "order_real{}".format(order_real)
