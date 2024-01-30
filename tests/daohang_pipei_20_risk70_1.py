# coding=utf-8
# -*- coding:utf-8 -*-
import pickle, sys
import pandas as pd
sys.path.append("../")

# from simulator.utilities import *
from alg_utility import *
from envs import *

from time import time as tttt

import networkx as nx
import graph
import numpy as np
distance=np.load('70distance20.npy')
crime=np.load('70crime20.npy')
#dijkstra requires a lot of space to execute
#well, its better than the time complexity
def dijkstra(ADT,start,end):
    """Dijkstra's Algorithm"""
    #all weights in dcg must be positive 
    #otherwise we have to use bellman ford instead
    # https://github.com/je-suis-tm/graph-theory/blob/master/bellman_ford%20forex%20arbitrage.ipynb
    neg_check=[j for i in ADT.reveal() for j in ADT.reveal()[i].values()]
    assert min(neg_check)>=0,"negative weights are not allowed, please use Bellman-Ford"
    
    #queue in dijkstra is similar to the queue in bfs
    #except when we pop items from queue
    #we pop the item with the minimum weight
    #thats why queue is a dict here rather than a list
    #ideally we should use priority queue
    queue={}
    queue[start]=0
    
    #distance is also a dictionary
    #it keeps track of distance from starting vertex to any vertex
    #before we start any iteration
    #we initialize all distances from start to any vertices to infinity
    #we set the distance[start] to zero
    distance={}
    for i in ADT.vertex():
        distance[i]=float('inf')
    distance[start]=0
        
    #pred is a dict as well
    #it keeps track of how we get to the current vertex
    #each time we update distance, we update the predecessor vertex
    #in the end, we can obtain the detailed path from start to end
    pred={}
        
    #for each iteration, we select a vertex with the minimum weight
    #we attempt to find the shortest path from the start to the current vertex
    #eventually we get the optimal result which is the shortest path
    #the logic is simple, similar to dynamic programming
    #if the path from start to end is optimal
    #the path from start to any vertex along the path 
    #should be the optimal as well
    #details about dynamic programming are in the following link
    # https://github.com/je-suis-tm/recursion-and-dynamic-programming/blob/master/knapsack.jl
    while queue:
        
        #vertex with the minimum weight in queue
        current=min(queue,key=queue.get)
        queue.pop(current)
        
        for j in ADT.edge(current):
            
            #check if the current vertex can construct the optimal path
            if distance[current]+ADT.weight(current,j)<distance[j]:
                distance[j]=distance[current]+ADT.weight(current,j)
                pred[j]=current
            
            #add child vertex to the queue
            if ADT.go(j)==0 and j not in queue:
                queue[j]=distance[j]
        
        #each vertex is visited only once
        ADT.visit(current)
        
        #traversal ends when the target is met
        if current==end:
            break
    
    #create the shortest path by backtracking
    #trace the predecessor vertex from end to start
    previous=end
    path=[]
    while pred:
        path.insert(0, previous)
        if previous==start:
            break
        previous=pred[previous]
    
    #note that if we cant go from start to end
    #we may get inf for distance[end]
    #additionally, the path may not include start position
    return distance[end],path
    
def A_star(G,Source,Goal):
    time1 = tttt()
    a=nx.astar_path ( G , Source , Goal , heuristic = None , weight = 'weight' )
    b=nx.astar_path_length ( G , Source , Goal , heuristic = None , weight = 'weight' )
    time2 = tttt()
    timecost = time2-time1
    print "搜索所用时间{}s".format(timecost)
    print "从节点'{}'到节点'{}' 所需要的经过的节点是 {} ,所需要的cost为 {}".format(Source,Goal,a,b)
    return a,timecost

def Dijkstra(ADT,Source,Goal):
    time1 = tttt()
    a,b = dijkstra(ADT,Source,Goal)
    time2 = tttt()
    timecost = time2-time1
    print "搜索所用时间{}s".format(timecost)
    print 'minimum steps:{}, path:{}'.format(a,b)
    return b,timecost

def cost(a):
    Dcost = []
    Ccost = []
    for i in range(len(a)-1):
        Dcost.append(distance[a[i]][a[i+1]])
        Ccost.append(crime[a[i]][a[i+1]])
    return Dcost,Ccost

# def find(which,G,Source,Goal):
#     if(which=='D'):
#         a = Dijkstra(G,Source,Goal)
#         Dcost,Ccost = cost(a)
#         print "路径上的距离代价为{}, 风险系数为{}".format(Dcost,Ccost)
#     if(which=='A'):
#         a = A_star(G,Source,Goal)
#         Dcost,Ccost = cost(a)
#         print "路径上的距离代价为{}, 风险系数为{}".format(Dcost,Ccost)

def find(mapped_matrix_int, a):
    M, N = mapped_matrix_int.shape
    for i in range(M):
        for j in range(N):
            if a == mapped_matrix_int[i][j]: return i, j

def find_du(i,j,n):
    x1 = i/n
    y1 = i%n
    x2 = j/n
    y2 = j%n
    du1 = x1-x2
    if du1 < 0: du1 = -du1
    du2 = y1-y2
    if du2 < 0: du2 = -du2
    du = du1 + du2
    return du


# 加入随机生成代码
import networkx as nx
import numpy as np
import random

# G = nx.Graph()
# distance = []
# crime = []
# for i in range(26):
#     G.add_node(i)
# for i in range(26):
#     D = []
#     C = []
#     for j in range(26):
#         if (j < i):
#             D.append(distance[j][i])
#             C.append(crime[j][i])
#         else:
#             if (i != j):
#                 flag = random.randint(1, 100)
#                 if flag < 135:
#                     d = random.randint(1, 90)#可能是变化幅度小了?
#                     c = random.randint(1, 50)
#                     D.append(d)
#                     C.append(c)
#                     G.add_edge(i, j, weight=d + c * 50)
#                 else:
#                     D.append(1)
#                     C.append(1)
#             else:
#                 D.append(1)
#                 C.append(1)
#     distance.append(D)
#     crime.append(C)
# distance = np.array(distance)
# crime = np.array(crime)

# print "distance{}".format(distance)

# print "crime{}".format(crime)

# np.save('distance.npy',distance)
# np.save('crime.npy',crime)

################## Load data ###################################
dir_prefix = ""
current_time = time.strftime("%Y%m%d_%H-%M")
log_dir = dir_prefix + "dispatch_simulator/experiments/{}/".format(current_time)
mkdir_p(log_dir)
print "log dir is {}".format(log_dir)

data_dir = dir_prefix + "dispatch_realdata/data_for_simulator/"
order_time_dist = []  # 原来是空的
order_price_dist = []
mapped_matrix_int = np.array([[5, 8, 2, 17, 15, 26, 27, 25, 30, 28, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109], [1, 7, 3, 23, 19, 29, 31, 33, 35, 37, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129],
 [6, 4, 9, 11, 22, 32, 34, 36, 38, 39, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119], [130, 140, 131, 141, 132, 142, 133, 143, 134, 144, 135, 145, 136, 146, 137, 147, 138, 148, 139, 149],
 [16, 10, 13, -100, 21, 40, 42, 44, 46, 48, 150, 152, 151, 154, 153, 156, 157, 158, 159, 155], [12, 14, 18, 20, 24, 41, 43, 45, 47, 49, 160, 162, 164, 161, 163, 166, 165, 168, 167, 169],
  [50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 171, 170, 174, 173, 178, 175, 176, 177, 179, 172], [181, 191, 180, 190, 182, 192, 183, 193, 184, 194, 185, 195, 187, 197, 186, 196, 189, 199, 188, 198],
  [51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 200, 201, 209, 203, 202, 204, 205, 206, 207, 208], [70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 210, 211, 212, 213, 219, 218, 217, 215, 214, 216],
  [71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 220, 221, 224, 223, 226, 225, 227, 229, 228, 222], [90, 92, 94, 96, 98, 91, 93, 95, 97, 99, 231, 239, 230, 238, 232, 233, 234, 236, 235, 237],
  [240, 250, 242, 252, 241, 251, 244, 254, 253, 243, 245, 256, 246, 247, 257, 248, 258, 249, 259, 255], [260, 270, 261, 271, 262, 272, 263, 273, 264, 274, 265, 275, 266, 276, 268, 279, 278, 269, 277, 267],
  [280, 290, 281, 291, 282, 292, 283, 293, 284, 294, 285, 295, 286, 296, 287, 297, 298, 288, 299, 289], [300, 302, 310, 312, 311, 301, 303, 313, 304, 314, 305, 315, 307, 317, 306, 316, 318, 309, 319, 308],
  [320, 321, 330, 331, 323, 333, 322, 332, 329, 339, 328, 338, 334, 324, 336, 326, 325, 335, 337, 327], [340, 342, 345, 350, 352, 355, 353, 343, 341, 351, 346, 356, 344, 354, 357, 347, 348, 358, 349, 359],
  [360, 370, 362, 372, 361, 371, 365, 375, 364, 374, 369, 379, 366, 373, 363, 376, 367, 377, 368, 378], [380, 390, 382, 392, 387, 397, 388, 398, 381, 391, 383, 393, 384, 394, 385, 395, 396, 386, 389, 399]
  ])

M, N = mapped_matrix_int.shape
order_num_dist = []
num_valid_grid = 399
idle_driver_location_mat = np.zeros((144, 399))


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

daoda = 399
chufa = 0
n1 = 400
m1 = 20
unvalid = 83
for i in range(n1):
    flag = 1
    if i != unvalid:
        x = i/m1
        y = i%m1
        if x-1>0:
            j = (x-1)*m1+y
            if j != unvalid:
                if crime[i][j] == 0: flag = 0
        if x+1<m1:
            j = (x+1)*m1+y
            if j != unvalid:
                if crime[i][j] == 0: flag = 0
        if y-1>0:
            j = x*m1+y-1
            if j != unvalid:
                if crime[i][j] == 0: flag = 0
        if y+1<m1:
            j = x*m1+y+1
            if j != unvalid:
                if crime[i][j] == 0: flag = 0
    if flag == 1: 
        chufa = i
        break

for x in range(n1):
    i = n1 - x - 1
    flag = 1
    if i != unvalid:
        x = i/m1
        y = i%m1
        if x-1>0:
            j = (x-1)*m1+y
            if j != unvalid:
                if crime[i][j] == 0: flag = 0
        if x+1<m1:
            j = (x+1)*m1+y
            if j != unvalid:
                if crime[i][j] == 0: flag = 0
        if y-1>0:
            j = x*m1+y-1
            if j != unvalid:
                if crime[i][j] == 0: flag = 0
        if y+1<m1:
            j = x*m1+y+1
            if j != unvalid:
                if crime[i][j] == 0: flag = 0
    if flag == 1: 
        daoda = i
        break

print "请输入您的到达点"
print "您的到达点为{}".format(daoda)

print "请输入您的出发点"
print "您的出发点为{}".format(chufa)


order_real = []
onoff_driver_location_mat = []
for tt in np.arange(144):
    for i in range(1, 400):
        a = []
        for j in range(1, 400):
            flag = random.randint(1, 10000)
            if flag < 10:
                if distance[i][j] != 0:
                    asd = find_du(i,j,20)
                    a.append([daoda, j, tt, 10, 1000+asd*30])
        order_real += a

        onoff_driver_location_mat.append([[-0.625,  2.92350389],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 3.36596622],
                                        [0.09090909, 8.46398452],
                                        [0.09090909, 3.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 2.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 3.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 4.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 7.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 6.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 1.46398452],
                                        [-0.625,       2.92350389],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 1.46398452],
                                        [-0.625,       2.92350389],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 1.46398452],
                                        [-0.625,       2.92350389],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 1.46398452],
                                        [-0.625,       2.92350389],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 3.36596622],
                                        [0.09090909, 8.46398452],
                                        [0.09090909, 3.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 2.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 3.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 4.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 7.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 6.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 1.46398452],
                                        [-0.625,       2.92350389],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 1.46398452],
                                        [-0.625,       2.92350389],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 1.46398452],
                                        [-0.625,       2.92350389],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 1.46398452],
                                        [-0.625,       2.92350389],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 3.36596622],
                                        [0.09090909, 8.46398452],
                                        [0.09090909, 3.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 2.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 3.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 4.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 7.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 6.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 1.46398452],
                                        [-0.625,       2.92350389],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 1.46398452],
                                        [-0.625,       2.92350389],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 1.46398452],
                                        [-0.625,       2.92350389],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 1.46398452],
                                        [-0.625,       2.92350389],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 3.36596622],
                                        [0.09090909, 8.46398452],
                                        [0.09090909, 3.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 2.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 3.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 4.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 7.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 6.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 1.46398452],
                                        [-0.625,       2.92350389],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 1.46398452],
                                        [-0.625,       2.92350389],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 1.46398452],
                                        [-0.625,       2.92350389],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  1.46398452]])
#print "order_real{}".format(order_real)


################## Initialize env ###################################
n_side = 6
GAMMA = 0.9


env = CityReal(mapped_matrix_int, order_num_dist, idle_driver_dist_time, idle_driver_location_mat,
                   order_time, order_price, l_max, M, N, n_side, 1, np.array(order_real), 
                   np.array(onoff_driver_location_mat))

state = env.reset_clean()

order_response_rates = []
T = 0
max_iter = 144
log_dir = dir_prefix + "dispatch_simulator/experiments/{}/".format(current_time)


temp = np.array(env.target_grids) + env.M * env.N
#print temp
target_id_states = env.target_grids + temp.tolist()
#print target_id_states
# curr_s = np.array(env.reset_clean()).flatten()  # [0] driver dist; [1] order dist
# curr_s = utility_conver_states(curr_s, target_id_states)
print "******************* Finish generating one day order **********************"



print "******************* Starting training Deep SARSA **********************"
from algorithm.IDQN import *

MAX_ITER = 50  # 10 iteration the Q-learning loss will converge.
is_plot_figure = False
city_time_start = 0
EP_LEN = 144
global_step = 0
city_time_end = city_time_start + EP_LEN
EPSILON = 0.9
gamma = 0.9
learning_rate = 1e-2

prev_epsiode_reward = 0
all_rewards = []
order_response_rates = []
value_table_sum = []
episode_rewards = []
episode_conflicts_drivers = []
order_response_rate_episode = []
episode_dispatched_drivers = []
T = 144
action_dim = 7
state_dim = env.n_valid_grids * 3 + T

# tf.reset_default_graph()
sess = tf.Session()
tf.set_random_seed(1)
q_estimator = Estimator(sess, action_dim,
                        state_dim,
                        env,
                        scope="q_estimator",
                        summaries_dir=log_dir)

target_estimator = Estimator(sess, action_dim, state_dim, env, scope="target_q")#初始化
sess.run(tf.global_variables_initializer())#不太清楚干了什么
estimator_copy = ModelParametersCopier(q_estimator, target_estimator)
replay = ReplayMemory(memory_size=1e+6, batch_size=3000)
stateprocessor = stateProcessor(target_id_states, env.target_grids, env.n_valid_grids)


data_save = []

saver = tf.train.Saver()
save_random_seed = []
N_ITER_RUNS = 25
temp_value = 10
RATIO = 1
EPSILON_start = 0.5
EPSILON_end = 0.1
epsilon_decay_steps=15
epsilons = np.linspace(EPSILON_start, EPSILON_end, epsilon_decay_steps)
for n_iter in np.arange(180):
    origin_distance=np.load('70distance20.npy')
    distance = origin_distance
    origin_crime=np.load('70crime20.npy')
    crime = origin_crime
    RANDOM_SEED = n_iter + MAX_ITER - temp_value
    env.reset_randomseed(RANDOM_SEED)
    save_random_seed.append(RANDOM_SEED)
    batch_s, batch_a, batch_r = [], [], []
    batch_reward_gmv = []
    epsiode_reward = 0
    num_dispatched_drivers = 0

    # reset env
    is_regenerate_order = 1
    curr_state = env.reset_clean(generate_order=is_regenerate_order, ratio=RATIO, city_time=city_time_start)
    # print "#coding=utf-8"
    # print curr_state[1]
    info = env.step_pre_order_assigin(curr_state)
    context = stateprocessor.compute_context(info)  # 剩余空闲司机的矩阵
    # print context
    curr_s = stateprocessor.utility_conver_states(curr_state)# curr_s是curr_state进行flatten操作以后，再去除非法节点的数据(example中第一个节点的driver和order数据)后的一维矩阵，在example中，curr_s是18-2=16个空间大小
    # print curr_s
    normalized_curr_s = stateprocessor.utility_normalize_states(curr_s)
    # print "normalized_curr_s"
    # print normalized_curr_s
    s_grid = stateprocessor.to_grid_states(normalized_curr_s, env.city_time)  # t0, s0
    # print "s_grid"
    # print s_grid
    # record rewards to update the value table
    episodes_immediate_rewards = []
    num_conflicts_drivers = []
    curr_num_actions = []
    epsilon = epsilons[n_iter] if n_iter < 15 else EPSILON_end

    startTime = tttt()
    route = []
    once_action = (0,chufa)
    route.append(once_action)

    rl_distance_cost = 0
    rl_crime_cost = 0  
    rl_distance_single_cost = []
    rl_crime_single_cost = []   

    dj_distance_cost = 0
    dj_crime_cost = 0  
    dj_crime_single_cost = []
    dj_distance_single_cost = []

    a_distance_cost = 0
    a_crime_cost = 0  
    a_crime_single_cost = []
    a_distance_single_cost = []

    axing_i = 0
    dj_i = 0
    rl_i = 0
    rl_end = 0
    axing = []
    dj = []
    rl_end_time = 0
    for ii in np.arange(EP_LEN + 1):
        # INPUT: state,  OUTPUT: action
        qvalues, action_idx, action_idx_valid, action_neighbor_idx, \
        action_tuple, action_starting_gridids = q_estimator.action(s_grid, context, epsilon) #Q是tf.sess
        # a0
        #print "action_tuple"
        #print action_tuple
        #print "{}iters {}  action_tuple {} ".format(n_iter,ii,action_tuple) 
        reward_cost = np.zeros(400)
        your_start_node = -1
        flag = 1
        number_max = -1
        start_node = route[-1][1]
        next_node = -1
        next_node = -1
        k = 1.1
        t = 1.3

        distance_parameter = 3
        crime_parameter = 10 - distance_parameter
        parameter = 0.1
        cost2_parameter = 2000
        daoda_parameter1 = 10000
        daoda_parameter2 = 100000
        risk_change_parameter1 = 1.00
        risk_change_parameter2 = 0.10
        
        if ii ==  0:
            startTime2 = tttt()
            G = nx.Graph()
            ADT=graph.graph()
            n = 400
            m = 20
            unvalid = 83
            for i in range(n):
                G.add_node(i)
            for i in range(n):
                if i != unvalid:
                    x = i/m
                    y = i%m
                    if x-1>0:
                        j = (x-1)*m+y
                        if j != unvalid:
                            G.add_edge(i, j, weight=distance[i][j]*distance_parameter+crime[i][j]*crime_parameter)
                            ADT.append(i, j, distance[i][j]*distance_parameter+crime[i][j]*crime_parameter)#构建边
                            ADT.append(j, i, distance[i][j]*distance_parameter+crime[i][j]*crime_parameter)#构建边
                    if x+1<m:
                        j = (x+1)*m+y
                        if j != unvalid:
                            G.add_edge(i, j, weight=distance[i][j]*distance_parameter+crime[i][j]*crime_parameter)
                            ADT.append(i, j, distance[i][j]*distance_parameter+crime[i][j]*crime_parameter)#构建边
                            ADT.append(j, i, distance[i][j]*distance_parameter+crime[i][j]*crime_parameter)#构建边
                    if y-1>0:
                        j = x*m+y-1
                        if j != unvalid:
                            G.add_edge(i, j, weight=distance[i][j]*distance_parameter+crime[i][j]*crime_parameter)
                            ADT.append(i, j, distance[i][j]*distance_parameter+crime[i][j]*crime_parameter)#构建边
                            ADT.append(j, i, distance[i][j]*distance_parameter+crime[i][j]*crime_parameter)#构建边
                    if y+1<m:
                        j = x*m+y+1
                        if j != unvalid:
                            G.add_edge(i, j, weight=distance[i][j]*distance_parameter+crime[i][j]*crime_parameter)
                            ADT.append(i, j, distance[i][j]*distance_parameter+crime[i][j]*crime_parameter)#构建边
                            ADT.append(j, i, distance[i][j]*distance_parameter+crime[i][j]*crime_parameter)#构建边
            print "A_star  -----  A_star"
            axing,ax_time = A_star(G,chufa,daoda)
            print "Dijkstra  -----  Dijkstra"
            dj,dj_time = Dijkstra(ADT,chufa,daoda)
            endTime2 = tttt()
            
        if rl_end == 0 or rl_end == 1:
            if rl_end == 1: rl_end = 2
            if len(route) - rl_i == 2:
                chushi = route[rl_i][1]
                mudi = route[rl_i+1][1]
                rl_distance_cost += distance[chushi][mudi]
                rl_crime_cost += crime[chushi][mudi]
                rl_distance_single_cost.append(distance[chushi][mudi])
                rl_crime_single_cost.append(crime[chushi][mudi])
                rl_i = rl_i + 1   
                if axing_i < len(axing)-1:
                    a_distance_cost += distance[axing[axing_i]][axing[axing_i+1]]
                    a_crime_cost += crime[axing[axing_i]][axing[axing_i+1]]
                    a_distance_single_cost.append(distance[axing[axing_i]][axing[axing_i+1]])
                    a_crime_single_cost.append(crime[axing[axing_i]][axing[axing_i+1]])
                    axing_i = axing_i + 1

                if dj_i < len(dj)-1:
                    dj_distance_cost += distance[dj[dj_i]][dj[dj_i+1]]
                    dj_crime_cost += crime[dj[dj_i]][dj[dj_i+1]]
                    dj_distance_single_cost.append(distance[dj[dj_i]][dj[dj_i+1]])
                    dj_crime_single_cost.append(crime[dj[dj_i]][dj[dj_i+1]]) 
                    dj_i = dj_i + 1
        
        if rl_end == 2:#如果RL已经结束了但是A*和迪杰斯特拉没结束
            if axing_i < len(axing)-1:
                a_distance_cost += distance[axing[axing_i]][axing[axing_i+1]]
                a_crime_cost += crime[axing[axing_i]][axing[axing_i+1]]
                a_distance_single_cost.append(distance[axing[axing_i]][axing[axing_i+1]])
                a_crime_single_cost.append(crime[axing[axing_i]][axing[axing_i+1]])
                axing_i = axing_i + 1

            if dj_i < len(dj)-1:
                dj_distance_cost += distance[dj[dj_i]][dj[dj_i+1]]
                dj_crime_cost += crime[dj[dj_i]][dj[dj_i+1]]
                dj_distance_single_cost.append(distance[dj[dj_i]][dj[dj_i+1]])
                dj_crime_single_cost.append(crime[dj[dj_i]][dj[dj_i+1]]) 
                dj_i = dj_i + 1       
            
        for node in action_tuple:
            start = node[0]
            end = node[1]
            number = node[2]
            cost1 = (0-distance[start][end]*number*distance_parameter-crime[start][end]*number*crime_parameter)
            cost2 = (find_du(start,daoda,20)-find_du(end,daoda,20))*cost2_parameter
            first_crime = crime[start][end]
            # crime[start][end] = 15000.0*pow(number,k-1)*pow(2.7,-number/t)/crime[start][end]/pow(t,k)
            crime[start][end] = (risk_change_parameter1/pow(number,risk_change_parameter2))*first_crime
            # if crime[start][end] >= first_crime*1.1:  #防止多的太多吧
            #     crime[start][end] = first_crime*1.1
            #if crime[start][end] <= first_crime*0.7:  #防止少的太多吧
            #    crime[start][end] = first_crime*0.7
            #if crime[start][end] < 10:
            #    crime[start][end] = 10
            #if crime[start][end] > 100:
            #    crime[start][end] = 100
            reward_cost[end] += cost1*(parameter)+cost2*(1-parameter)
            if rl_end == 0:
                if ii >= 30:
                    if start == start_node: #这一段改进缩短了很多
                        tianjia = 1
                        for xx in route:
                            if end == xx[1]: tianjia = 0
                        if number > number_max and tianjia == 1:
                            number_max = number
                            next_node = end 
                    if start != start_node and number_max > 0 and flag != 3:#判断有路线而且过了出发点判定  
                        once_action = (ii,next_node)
                        route.append(once_action)
                        #print "加了一个"
                        flag = 3#结束了
                        if next_node == daoda:
                            reward_cost[end] += daoda_parameter1 + daoda_parameter2*find_du(chufa,daoda,20)
                            rl_end = 1
                            rl_end_time = tttt()



        # ONE STEP: r0
        next_state, r, info = env.step(reward_cost,action_tuple, 2)# next_state = self.get_observation()，这里的next_state就是剩余空闲driver和剩余订单两个array

        #新加的在step里的info里

        #print "info"
        #print info

        # r0
        immediate_reward = stateprocessor.reward_wrapper(info, curr_s)#每个node的reward/driver数量获得一个平均值

        #print "immediate_reward"
        #print immediate_reward

        # a0
        action_mat = stateprocessor.to_action_mat(action_neighbor_idx)

        # s0
        #print "s_grid"
        #print s_grid
        #这里已经发生了变化，s_grid_train的N，已经变成了转移driver的数量
        if(np.sum(action_starting_gridids)==0):
            
            break;

        s_grid_train = stateprocessor.to_grid_state_for_training(s_grid, action_starting_gridids)#action_starting_gridids这个应该是依次储存了每个点的driver转移的数量
        #print "s_grid_train"
        #print s_grid_train.shape
        # s1
        s_grid_next = stateprocessor.to_grid_next_states(s_grid_train, next_state, action_idx_valid, env.city_time)#action_idx_valid应该是依次储存了各个action 的合法的the index in nodes

        # Save transition to replay memory
        if ii != 0:
            # r1, c0
            r_grid = stateprocessor.to_grid_rewards(action_idx_valid_prev, immediate_reward)
            targets_batch = r_grid + gamma * target_estimator.predict(s_grid_next_prev)   #r+衰减系数哈哈

            # s0, a0, r1
            replay.add(state_mat_prev, action_mat_prev, targets_batch, s_grid_next_prev)

        state_mat_prev = s_grid_train
        action_mat_prev = action_mat
        context_prev = context
        s_grid_next_prev = s_grid_next
        action_idx_valid_prev = action_idx_valid

        # c1
        context = stateprocessor.compute_context(info[1])
        # s1
        curr_s = stateprocessor.utility_conver_states(next_state)
        normalized_curr_s = stateprocessor.utility_normalize_states(curr_s)
        s_grid = stateprocessor.to_grid_states(normalized_curr_s, env.city_time)  # t0, s0
        # Sample a minibatch f rom the replay memory and update q network training method1
        if replay.curr_lens != 0:
            for _ in np.arange(20):
                fetched_batch = replay.sample()
                mini_s, mini_a, mini_target, mini_next_s = fetched_batch
                q_estimator.update(mini_s, mini_a, mini_target, learning_rate, global_step)
                global_step += 1

        # Perform gradient descent update
        # book keeping
        global_step += 1
        all_rewards.append(r)
        batch_reward_gmv.append(r)
        order_response_rates.append(env.order_response_rate)
        num_conflicts_drivers.append(collision_action(action_tuple))
        curr_num_action = np.sum([aa[2] for aa in action_tuple]) if len(action_tuple) != 0 else 0
        curr_num_actions.append(curr_num_action)


    duration = tttt() - startTime - (endTime2-startTime2)
    rl_duration = (rl_end_time - startTime) - (endTime2-startTime2)

        
    print("Times takes:",duration)
    #crime = crime + 10
    # distance_cost = 0
    # crime_cost = 0  
    # crime_single_cost = []
    # for i in range(len(route)-1):
    #     chushi = route[i][1]
    #     mudi = route[i+1][1]
    #     distance_cost += distance[chushi][mudi]
    #     crime_cost += crime[chushi][mudi]
    #     crime_single_cost.append(crime[chushi][mudi])
    
    print "Dijkstra  -----  Dijkstra"
    print "第{}天,您的路线为{}".format(n_iter,dj)
    print "您的距离系数损耗为{},风险系数损耗为{},每步距离损耗{},每步风险损耗{}".format(dj_distance_cost,dj_crime_cost,dj_distance_single_cost,dj_crime_single_cost)

    print "A_star  -----  A_star"
    print "第{}天,您的路线为{}".format(n_iter,axing)
    print "您的距离系数损耗为{},风险系数损耗为{},每步距离损耗{},每步风险损耗{}".format(a_distance_cost,a_crime_cost,a_distance_single_cost,a_crime_single_cost)

    print "RL  -----  RL"
    print "第{}天,您的路线为{}".format(n_iter,route)
    print "您的距离系数损耗为{},风险系数损耗为{},每步距离损耗{},每步风险损耗{}".format(rl_distance_cost,rl_crime_cost,rl_distance_single_cost,rl_crime_single_cost)
    # print "输入任意数字以确认"
    # suiji = input()
    
    one_data = [dj_distance_cost,a_distance_cost,rl_distance_cost,dj_crime_cost,a_crime_cost,rl_crime_cost,
    dj_distance_single_cost,a_distance_single_cost,rl_distance_single_cost,dj_crime_single_cost,a_crime_single_cost,rl_crime_single_cost,
    dj,axing,route,dj_time,ax_time,duration,rl_duration]

    data_save.append(one_data)

    episode_reward = np.sum(batch_reward_gmv[1:])
    episode_rewards.append(episode_reward)
    n_iter_order_response_rate = np.mean(order_response_rates[1:])
    order_response_rate_episode.append(n_iter_order_response_rate)
    episode_conflicts_drivers.append(np.sum(num_conflicts_drivers[:-1]))
    episode_dispatched_drivers.append(np.sum(curr_num_actions[:-1]))

    print "iteration {} ********* reward {} order{} conflicts {} drivers {}".format(n_iter, episode_reward,
                                                                                             order_response_rate_episode[-1],
                                                                                             episode_conflicts_drivers[-1],
                                                                                             episode_dispatched_drivers[-1])

    pickle.dump([episode_rewards, order_response_rate_episode, save_random_seed, episode_conflicts_drivers,
                 episode_dispatched_drivers], open(log_dir + "results.pkl", "w"))

    if n_iter == 180:
        break

    # # training method 2.
    # for _ in np.arange(4000):
    #     fetched_batch = replay.sample()
    #     mini_s, mini_a, mini_target, mini_next_s = fetched_batch
    #     q_estimator.update(mini_s, mini_a, mini_target, learning_rate, global_step)
    #     global_step += 1

    # update target Q network
    estimator_copy.make(sess)


    saver.save(sess, log_dir+"model.ckpt")


data=pd.DataFrame(data_save,
                   columns=('dj_distance_cost','a_distance_cost','rl_distance_cost','dj_crime_cost','a_crime_cost','rl_crime_cost',
    'dj_distance_single_cost','a_distance_single_cost','rl_distance_single_cost','dj_crime_single_cost','a_crime_single_cost','rl_crime_single_cost',
    'dj','axing','route','dj_time','ax_time','oneday_duration','rl_duration'))

data.to_excel('70risk_20_1_data1.00_0.10_37.xls')
