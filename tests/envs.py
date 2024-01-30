# coding=utf-8

import os, sys, random, time
import logging

sys.path.append("../")

from objects import *
from utilities import *

# from algorithm import *

# current_time = time.strftime("%Y%m%d_%H-%M")
# log_dir = "/nfs/private/linkaixiang_i/data/dispatch_simulator/experiments/"+current_time + "/"
# mkdir_p(log_dir)
# logging.basicConfig(filename=log_dir +'logger_env.log', level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger_ch = logging.StreamHandler()
logger_ch.setLevel(logging.DEBUG)
logger_ch.setFormatter(logging.Formatter(
    '%(asctime)s[%(levelname)s][%(lineno)s:%(funcName)s]||%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(logger_ch)
RANDOM_SEED = 0  # unit test use this random seed.


class CityReal:
    '''A real city is consists of M*N grids '''

    def __init__(self, mapped_matrix_int, order_num_dist, idle_driver_dist_time, idle_driver_location_mat,
                 order_time_dist, order_price_dist,
                 l_max, M, N, n_side, probability=1.0 / 28, real_orders="", onoff_driver_location_mat="",
                 global_flag="global", time_interval=10):
        """
        :param mapped_matrix_int: 2D matrix: each position is either -100 or grid id from order in real data.
        :param order_num_dist: 144 [{node_id1: [mu, std]}, {node_id2: [mu, std]}, ..., {node_idn: [mu, std]}]
                            node_id1 is node the index in self.nodes
        :param idle_driver_dist_time: [[mu1, std1], [mu2, std2], ..., [mu144, std144]] mean and variance of idle drivers in
        the city at each time
        :param idle_driver_location_mat: 144 x num_valid_grids matrix.
        :param order_time_dist: [ 0.27380797,..., 0.00205766] The probs of order duration = 1 to 9
        :param order_price_dist: [[10.17, 3.34],   # mean and std of order's price, order durations = 10 minutes.
                                   [15.02, 6.90],  # mean and std of order's price, order durations = 20 minutes.
                                   ...,]
        :param onoff_driver_location_mat: 144 x 504 x 2: 144 total time steps, num_valid_grids = 504.
        mean and std of online driver number - offline driver number
        onoff_driver_location_mat[t] = [[-0.625       2.92350389]  <-- Corresponds to the grid in target_node_ids
                                        [ 0.09090909  1.46398452]
                                        [ 0.09090909  2.36596622]
                                        [-1.2         2.05588586]...]
        :param M:
        :param N:
        :param n_side:
        :param time_interval:
        :param l_max: The max-duration of an order
        :return:
        """
        # City.__init__(self, M, N, n_side, time_interval)
        self.M = M  # row numbers
        self.N = N  # column numbers
        self.nodes = [Node(i) for i in xrange(M * N)]  # a list of nodes: node id start from 0
        self.drivers = {}  # driver[driver_id] = driver_instance  , driver_id start from 0
        self.n_drivers = 0  # total idle number of drivers. online and not on service.
        self.n_offline_drivers = 0  # total number of offline drivers.
        self.construct_map_simulation(M, N, n_side)
        self.city_time = 0
        # self.idle_driver_distribution = np.zeros((M, N))
        self.n_intervals = 1440 / time_interval
        self.n_nodes = self.M * self.N
        self.n_side = n_side
        self.order_response_rate = 0

        self.RANDOM_SEED = RANDOM_SEED

        self.l_max = l_max  # Start from 1. The max number of layers an order can across.
        assert l_max <= M - 1 and l_max <= N - 1
        assert 1 <= l_max <= 9  # Ignore orders less than 10 minutes and larger than 1.5 hours

        self.target_grids = []
        self.n_valid_grids = 0  # num of valid grid
        self.nodes = [None for _ in np.arange(self.M * self.N)]
        self.construct_node_real(mapped_matrix_int)  # target_grids=[0, 2, 3, 4, 5, 6, 7, 8]
        self.mapped_matrix_int = mapped_matrix_int  # mapped_matrix_int = np.array([[1, -100, 3], [5, 4, 2], [6, 7, 8]])

        self.construct_map_real(n_side)  # 设置邻居节点(根据点的分布和n_side)
        self.order_num_dist = order_num_dist  # 储存了每个时间点每个node订单数量的mean和std
        self.distribution_name = "Poisson"
        self.idle_driver_dist_time = idle_driver_dist_time  # 储存了每个时间点的idle_driver数量的mean和std
        self.idle_driver_location_mat = idle_driver_location_mat  # 储存了每个时间点的每个node的idle_driver数量的mean

        self.order_time_dist = order_time_dist[:l_max] / np.sum(order_time_dist[:l_max])  # example没有设置
        self.order_price_dist = order_price_dist  # 没有设置

        target_node_ids = []
        target_grids_sorted = np.sort(
            mapped_matrix_int[np.where(mapped_matrix_int > 0)])  # 把valid的点顺序排序 [1 2 3 4 5 6 7 8]
        # print "target_grids_sorted"
        # print target_grids_sorted
        for item in target_grids_sorted:
            x, y = np.where(mapped_matrix_int == item)
            target_node_ids.append(ids_2dto1d(x, y, M, N)[0])
        self.target_node_ids = target_node_ids  # [0, 5, 2, 4, 3, 6, 7, 8]记录的是第N个点的位次，例如第一个是array中0位，第二个在第5位。这里第N个，输入的真实node的序号排序下来的第N个
        # print "target_node_ids"
        # print self.target_node_ids
        # store valid note id. Sort by number of orders emerged. descending.

        self.node_mapping = {}
        self.construct_mapping()  # node_mapping储存的是输入的真实node的id对应的在mapped_matrix_int矩阵中的位次

        self.real_orders = real_orders  # 4 weeks' data
        # [[92, 300, 143, 2, 13.2],...] origin grid, destination grid, start time, end time, price.

        self.p = probability  # sample probability
        self.time_keys = [int(dt.strftime('%H%M')) for dt in
                          datetime_range(datetime(2017, 9, 1, 0), datetime(2017, 9, 2, 0),
                                         timedelta(minutes=time_interval))]
        self.day_orders = []  # one day's order.

        self.onoff_driver_location_mat = onoff_driver_location_mat

        # Stats
        self.all_grids_on_number = 0  # current online # drivers.
        self.all_grids_off_number = 0

        self.out_grid_in_orders = np.zeros((self.n_intervals, len(self.target_grids)))
        self.global_flag = global_flag
        self.weights_layers_neighbors = [1.0, np.exp(-1), np.exp(-2)]

    def construct_map_simulation(self, M, N, n):
        """Connect node to its neighbors based on a simulated M by N map
            :param M: M row index matrix
            :param N: N column index matrix
            :param n: n - sided polygon
        """
        for idx, current_node in enumerate(self.nodes):
            if current_node is not None:
                i, j = ids_1dto2d(idx, M, N)
                current_node.set_neighbors(get_neighbor_list(i, j, M, N, n, self.nodes))
                # print current_node.layers_neighbors_id

    def construct_mapping(self):
        """
        :return:
        """
        target_grid_id = self.mapped_matrix_int[
            np.where(self.mapped_matrix_int > 0)]  # target_grid_id=[1 3 5 4 2 6 7 8]
        # print "target_grid_id"
        # print target_grid_id
        for g_id, n_id in zip(target_grid_id, self.target_grids):  # target_grids=[0, 2, 3, 4, 5, 6, 7, 8]
            self.node_mapping[g_id] = n_id  # node_mapping储存的是输入的真实node的id对应的在mapped_matrix_int矩阵中的位次

    def construct_node_real(self, mapped_matrix_int):
        """ Initialize node, only valid node in mapped_matrix_in will be initialized.
        """
        row_inds, col_inds = np.where(mapped_matrix_int >= 0)  # row_ind表示一个array，包含每个合法点的行号，col_inds包含每个合法点的列号
        # print np.where(mapped_matrix_int >= 0)
        # print"row_inds"
        # print row_inds
        # print col_inds
        target_ids = []  # start from 0. 
        for x, y in zip(row_inds, col_inds):
            node_id = ids_2dto1d(x, y, self.M, self.N)
            # 在这时,遍历中-100的非法点已经没有参与其中
            # print "node_id"
            # print node_id
            self.nodes[node_id] = Node(node_id)  # node id start from 0.
            target_ids.append(node_id)

        for x, y in zip(row_inds, col_inds):
            node_id = ids_2dto1d(x, y, self.M, self.N)
            self.nodes[node_id].get_layers_neighbors(self.l_max, self.M, self.N, self)

        self.target_grids = target_ids  # example中是[0, 2, 3, 4, 5, 6, 7, 8]
        self.n_valid_grids = len(target_ids)
        # print "target_grids"
        # print self.target_grids

    def construct_map_real(self, n_side):
        """Build node connection. 
        """
        for idx, current_node in enumerate(self.nodes):
            i, j = ids_1dto2d(idx, self.M, self.N)
            if current_node is not None:
                current_node.set_neighbors(get_neighbor_list(i, j, self.M, self.N, n_side, self.nodes))  # 设置邻居节点

    def initial_order_random(self, distribution_all, dis_paras_all):
        """ Initialize order distribution
        :param distribution: 'Poisson', 'Gaussian'
        :param dis_paras:     lambda,    mu, sigma
        """
        for idx, node in enumerate(self.nodes):
            if node is not None:
                node.order_distribution(distribution_all[idx], dis_paras_all[idx])

    def get_observation(self):
        next_state = np.zeros((2, self.M, self.N))
        for _node in self.nodes:
            if _node is not None:
                row_id, column_id = ids_1dto2d(_node.get_node_index(), self.M, self.N)
                next_state[0, row_id, column_id] = _node.idle_driver_num
                next_state[1, row_id, column_id] = _node.order_num

        return next_state

    def get_num_idle_drivers(self):
        """ Compute idle drivers
        :return:
        """
        temp_n_idle_drivers = 0
        for _node in self.nodes:
            if _node is not None:
                temp_n_idle_drivers += _node.idle_driver_num
        return temp_n_idle_drivers

    def get_observation_driver_state(self):
        """ Get idle driver distribution, computing #drivers from node.
        :return:
        """
        next_state = np.zeros((self.M, self.N))
        for _node in self.nodes:
            if _node is not None:
                row_id, column_id = ids_1dto2d(_node.get_node_index(), self.M, self.N)
                next_state[row_id, column_id] = _node.get_idle_driver_numbers_loop()

        return next_state

    def reset_randomseed(self, random_seed):
        self.RANDOM_SEED = random_seed

    def reset(self):
        """ Return initial observation: get order distribution and idle driver distribution

        """

        _M = self.M
        _N = self.N
        assert self.city_time == 0
        # initialization drivers according to the distribution at time 0
        num_idle_driver = self.utility_get_n_idle_drivers_real()
        self.step_driver_online_offline_control(num_idle_driver)

        # generate orders at first time step
        distribution_name = [self.distribution_name] * (_M * _N)
        distribution_param_dictionary = self.order_num_dist[self.city_time]
        distribution_param = [0] * (_M * _N)
        for key, value in distribution_param_dictionary.iteritems():
            if self.distribution_name == 'Gaussian':
                mu, sigma = value
                distribution_param[key] = mu, sigma
            elif self.distribution_name == 'Poisson':
                mu = value[0]
                distribution_param[key] = mu
            else:
                print "Wrong distribution"

        self.initial_order_random(distribution_name, distribution_param)
        self.step_generate_order_real()

        return self.get_observation()

    def reset_clean(self, generate_order=1, ratio=1, city_time=""):
        """ 1. bootstrap oneday's order data.
            2. clean current drivers and orders, regenerate new orders and drivers.
            can reset anytime
        :return:
        """
        if city_time != "":
            self.city_time = city_time

        # clean orders and drivers
        self.drivers = {}  # driver[driver_id] = driver_instance  , driver_id start from 0
        self.n_drivers = 0  # total idle number of drivers. online and not on service.
        self.n_offline_drivers = 0  # total number of offline drivers.
        for node in self.nodes:
            if node is not None:
                node.clean_node()

        # Generate one day's order.
        if generate_order == 1:
            self.utility_bootstrap_oneday_order()

        # Init orders of current time step
        moment = self.city_time % self.n_intervals
        self.step_bootstrap_order_real(self.day_orders[moment])

        # Init current driver distribution
        if self.global_flag == "global":
            num_idle_driver = self.utility_get_n_idle_drivers_real()
            num_idle_driver = int(num_idle_driver * ratio)
        else:
            num_idle_driver = self.utility_get_n_idle_drivers_nodewise()
        self.step_driver_online_offline_control_new(num_idle_driver)
        return self.get_observation()

    def utility_collect_offline_drivers_id(self):
        """count how many drivers are offline
        :return: offline_drivers: a list of offline driver id
        """
        count = 0  # offline driver num
        offline_drivers = []  # record offline driver id
        for key, _driver in self.drivers.iteritems():
            if _driver.online is False:
                count += 1
                offline_drivers.append(_driver.get_driver_id())
        return offline_drivers

    def utility_get_n_idle_drivers_nodewise(self):
        """ compute idle drivers.
        :return:
        """
        time = self.city_time % self.n_intervals
        idle_driver_num = np.sum(self.idle_driver_location_mat[time])
        return int(idle_driver_num)

    def utility_add_driver_real_new(self, num_added_driver):
        curr_idle_driver_distribution = self.get_observation()[0]
        curr_idle_driver_distribution_resort = np.array(
            [int(curr_idle_driver_distribution.flatten()[index]) for index in
             self.target_node_ids])

        idle_driver_distribution = self.idle_driver_location_mat[self.city_time % self.n_intervals, :]

        idle_diff = idle_driver_distribution.astype(int) - curr_idle_driver_distribution_resort
        idle_diff[np.where(idle_diff <= 0)] = 0

        node_ids = np.random.choice(self.target_node_ids, size=[num_added_driver],
                                    p=idle_diff / float(np.sum(idle_diff)))

        n_total_drivers = len(self.drivers.keys())
        for ii, node_id in enumerate(node_ids):
            added_driver_id = n_total_drivers + ii
            self.drivers[added_driver_id] = Driver(added_driver_id)
            self.drivers[added_driver_id].set_position(self.nodes[node_id])
            self.nodes[node_id].add_driver(added_driver_id, self.drivers[added_driver_id])

        self.n_drivers += num_added_driver

    def utility_add_driver_real_new_offlinefirst(self, num_added_driver):

        # curr_idle_driver_distribution = self.get_observation()[0][np.where(self.mapped_matrix_int > 0)]
        curr_idle_driver_distribution = self.get_observation()[0]
        curr_idle_driver_distribution_resort = np.array(
            [int(curr_idle_driver_distribution.flatten()[index]) for index in
             self.target_node_ids])

        idle_driver_distribution = self.idle_driver_location_mat[self.city_time % self.n_intervals, :]

        idle_diff = idle_driver_distribution.astype(int) - curr_idle_driver_distribution_resort
        idle_diff[np.where(idle_diff <= 0)] = 0

        if float(np.sum(idle_diff)) == 0:
            return
        np.random.seed(self.RANDOM_SEED)
        node_ids = np.random.choice(self.target_node_ids, size=[num_added_driver],
                                    p=idle_diff / float(np.sum(idle_diff)))

        for ii, node_id in enumerate(node_ids):

            if self.nodes[node_id].offline_driver_num > 0:
                self.nodes[node_id].set_offline_driver_online()
                self.n_drivers += 1
                self.n_offline_drivers -= 1
            else:

                n_total_drivers = len(self.drivers.keys())
                added_driver_id = n_total_drivers
                self.drivers[added_driver_id] = Driver(added_driver_id)
                self.drivers[added_driver_id].set_position(self.nodes[node_id])
                self.nodes[node_id].add_driver(added_driver_id, self.drivers[added_driver_id])
                self.n_drivers += 1

    def utility_add_driver_real_nodewise(self, node_id,
                                         num_added_driver):  # 在每个时刻一开始的作用：num_added_driver是根据输入的mean和std通过正态分布随机取得的值，当值>0是，会用到此函数，为当前node增加idle_driver

        # 如果当前点有下线的司机，会改变状态为在线，如果当前点的离线司机不够时，会新增新的driver
        while num_added_driver > 0:
            if self.nodes[node_id].offline_driver_num > 0:
                self.nodes[node_id].set_offline_driver_online()  # 这里应该是外来订单在该时刻完成后，司机在该终点node变成了空闲driver
                self.n_drivers += 1
                self.n_offline_drivers -= 1
            else:

                n_total_drivers = len(self.drivers.keys())  # 为外来司机设置id
                added_driver_id = n_total_drivers
                self.drivers[added_driver_id] = Driver(added_driver_id)
                self.drivers[added_driver_id].set_position(self.nodes[node_id])
                self.nodes[node_id].add_driver(added_driver_id, self.drivers[added_driver_id])
                self.n_drivers += 1
            num_added_driver -= 1

    def utility_set_drivers_offline_real_nodewise(self, node_id, n_drivers_to_off):

        while n_drivers_to_off > 0:
            if self.nodes[node_id].idle_driver_num > 0:
                self.nodes[node_id].set_idle_driver_offline_random()
                self.n_drivers -= 1
                self.n_offline_drivers += 1
                n_drivers_to_off -= 1
                self.all_grids_off_number += 1
            else:
                break

    def utility_set_drivers_offline_real_new(self, n_drivers_to_off):

        curr_idle_driver_distribution = self.get_observation()[0]
        curr_idle_driver_distribution_resort = np.array([int(curr_idle_driver_distribution.flatten()[index])
                                                         for index in self.target_node_ids])

        # historical idle driver distribution
        idle_driver_distribution = self.idle_driver_location_mat[self.city_time % self.n_intervals, :]

        # diff of curr idle driver distribution and history
        idle_diff = curr_idle_driver_distribution_resort - idle_driver_distribution.astype(int)
        idle_diff[np.where(idle_diff <= 0)] = 0

        n_drivers_can_be_off = int(np.sum(curr_idle_driver_distribution_resort[np.where(idle_diff >= 0)]))
        if n_drivers_to_off > n_drivers_can_be_off:
            n_drivers_to_off = n_drivers_can_be_off

        sum_idle_diff = np.sum(idle_diff)
        if sum_idle_diff == 0:
            return
        np.random.seed(self.RANDOM_SEED)
        node_ids = np.random.choice(self.target_node_ids, size=[n_drivers_to_off],
                                    p=idle_diff / float(sum_idle_diff))

        for ii, node_id in enumerate(node_ids):
            if self.nodes[node_id].idle_driver_num > 0:
                self.nodes[node_id].set_idle_driver_offline_random()
                self.n_drivers -= 1
                self.n_offline_drivers += 1
                n_drivers_to_off -= 1

    def utility_bootstrap_oneday_order(self):

        num_all_orders = len(self.real_orders)
        index_sampled_orders = np.where(
            np.random.binomial(1, self.p, num_all_orders) == 1)  # 默认p=1/28.0，但是在example里是1，也就是
        # print type(index_sampled_orders)
        # print index_sampled_orders
        one_day_orders = self.real_orders[index_sampled_orders]
        # print one_day_orders
        self.out_grid_in_orders = np.zeros((self.n_intervals, len(self.target_grids)))  # 外面的网格进来的订单

        day_orders = [[] for _ in np.arange(self.n_intervals)]
        for iorder in one_day_orders:
            #  iorder: [92, 300, 143, 2, 13.2]
            start_time = int(iorder[2])
            if iorder[0] not in self.node_mapping.keys() and iorder[1] not in self.node_mapping.keys():
                continue
            start_node = self.node_mapping.get(iorder[0], -100)
            end_node = self.node_mapping.get(iorder[1], -100)
            duration = int(iorder[3])
            price = iorder[4]

            if start_node == -100:
                column_index = self.target_grids.index(end_node)
                self.out_grid_in_orders[(start_time + duration) % self.n_intervals, column_index] += 1  # 外来订单
                continue

            day_orders[start_time].append([start_node, end_node, start_time, duration, price])
        self.day_orders = day_orders

    def step_driver_status_control(self):
        # Deal with orders finished at time T=1, check driver status. finish order, set back to off service
        for key, _driver in self.drivers.iteritems():
            _driver.status_control_eachtime(self)  # 该函数拿来计算每个订单的结束时间，与现在时间比较，在此时完成的订单会被完成，然后设置driver为off_service
        moment = self.city_time % self.n_intervals
        orders_to_on_drivers = self.out_grid_in_orders[moment, :]  # 外面的网格进来的订单
        # print "self.out_grid_in_orders[moment, :]"
        # print self.out_grid_in_orders[moment, :]
        # print "orders_to_on_drivers"
        # print orders_to_on_drivers
        for idx, item in enumerate(orders_to_on_drivers):  # idx表示第几个node,item表示数量
            # print idx
            # print item
            if item != 0:
                node_id = self.target_grids[idx]
                self.utility_add_driver_real_nodewise(node_id, int(item))  # 为外来订单的操作，为目的地增加了idle_driver,但是没有得reward

    def step_driver_online_offline_nodewise(self):
        """ node wise control driver online offline
        :return:
        """
        moment = self.city_time % self.n_intervals
        curr_onoff_distribution = self.onoff_driver_location_mat[moment]

        self.all_grids_on_number = 0
        self.all_grids_off_number = 0
        for idx, target_node_id in enumerate(self.target_node_ids):  # target_node_ids好像只有合法点
            curr_mu = curr_onoff_distribution[idx, 0]
            curr_sigma = curr_onoff_distribution[idx, 1]
            on_off_number = np.round(np.random.normal(curr_mu, curr_sigma, 1)[0]).astype(int)
            # print "on_off_number"
            # print on_off_number
            if on_off_number > 0:
                self.utility_add_driver_real_nodewise(target_node_id, on_off_number)
                self.all_grids_on_number += on_off_number
            elif on_off_number < 0:
                self.utility_set_drivers_offline_real_nodewise(target_node_id,
                                                               abs(on_off_number))  # on_off_number<0时，让司机随机下线，数量为该值，如果当前空闲driver数量低于该值，则到0，不再下线
            else:
                pass

    def step_driver_online_offline_control_new(self, n_idle_drivers):
        """ control the online offline status of drivers

        :param n_idle_drivers: the number of idle drivers expected at current moment
        :return:
        """

        offline_drivers = self.utility_collect_offline_drivers_id()
        self.n_offline_drivers = len(offline_drivers)

        if n_idle_drivers > self.n_drivers:

            self.utility_add_driver_real_new_offlinefirst(n_idle_drivers - self.n_drivers)  # 如果当前司机数量不足，增加空闲司机数量

        elif n_idle_drivers < self.n_drivers:
            self.utility_set_drivers_offline_real_new(self.n_drivers - n_idle_drivers)  # 相反
        else:
            pass

    def step_driver_online_offline_control(self, n_idle_drivers):
        """ control the online offline status of drivers

        :param n_idle_drivers: the number of idle drivers expected at current moment
        :return:
        """

        offline_drivers = self.utility_collect_offline_drivers_id()
        self.n_offline_drivers = len(offline_drivers)
        if n_idle_drivers > self.n_drivers:
            # bring drivers online.
            while self.n_drivers < n_idle_drivers:
                if self.n_offline_drivers > 0:
                    for ii in np.arange(self.n_offline_drivers):
                        self.drivers[offline_drivers[ii]].set_online()
                        self.n_drivers += 1
                        self.n_offline_drivers -= 1
                        if self.n_drivers == n_idle_drivers:
                            break

                self.utility_add_driver_real_new(n_idle_drivers - self.n_drivers)

        elif n_idle_drivers < self.n_drivers:
            self.utility_set_drivers_offline_real_new(self.n_drivers - n_idle_drivers)
        else:
            pass

    def utility_get_n_idle_drivers_real(self):
        """ control the number of idle drivers in simulator;
        :return:
        """
        time = self.city_time % self.n_intervals
        mean, std = self.idle_driver_dist_time[time]
        np.random.seed(self.city_time)
        return np.round(np.random.normal(mean, std, 1)[0]).astype(int)  # 正态分布随机取一个driver数量

    def utility_set_neighbor_weight(self, weights):
        self.weights_layers_neighbors = weights

    def step_generate_order_real(self):
        # generate order at t + 1
        for node in self.nodes:
            if node is not None:
                node_id = node.get_node_index()
                # generate orders start from each node
                random_seed = node.get_node_index() + self.city_time
                node.generate_order_real(self.l_max, self.order_time_dist, self.order_price_dist,
                                         self.city_time, self.nodes, random_seed)

    def step_bootstrap_order_real(self, day_orders_t):
        for iorder in day_orders_t:
            start_node_id = iorder[0]
            end_node_id = iorder[1]
            start_node = self.nodes[start_node_id]

            if end_node_id in self.target_grids:
                end_node = self.nodes[end_node_id]
            else:
                end_node = None  # 到地图外面去了？
            start_node.add_order_real(self.city_time, end_node, iorder[3], iorder[4])

    def step_assign_order(self):

        reward = 0  # R_{t+1}
        all_order_num = 0
        finished_order_num = 0
        for node in self.nodes:
            if node is not None:
                node.remove_unfinished_order(self.city_time)
                reward_node, all_order_num_node, finished_order_num_node = node.simple_order_assign_real(self.city_time,
                                                                                                         self)
                reward += reward_node
                all_order_num += all_order_num_node
                finished_order_num += finished_order_num_node
        if all_order_num != 0:
            self.order_response_rate = finished_order_num / float(all_order_num)
        else:
            self.order_response_rate = -1
        return reward

    def step_assign_order_broadcast_neighbor_reward_update(self, reward_cost):
        """ Consider the orders whose destination or origin is not in the target region
        :param num_layers:
        :param weights_layers_neighbors: [1, 0.5, 0.25, 0.125]
        :return:
        """

        node_reward = np.zeros((len(self.nodes)))
        neighbor_reward = np.zeros((len(self.nodes)))
        # First round broadcast
        reward = 0  # R_{t+1}
        all_order_num = 0
        finished_order_num = 0
        for node in self.nodes:
            if node is not None:
                reward_node, all_order_num_node, finished_order_num_node = node.simple_order_assign_real(self.city_time,
                                                                                                         self)
                reward += reward_node
                #print "reward"
                #print reward
                all_order_num += all_order_num_node
                finished_order_num += finished_order_num_node
                node_reward[node.get_node_index()] += reward_node + reward_cost[node.get_node_index()] #加的额外奖惩 如果不加get_node_index是存储地址
        # Second round broadcast

#        for node in self.nodes:  # 第二轮应该是要把剩余的待分配订单分配到邻节点的闲置司机上
#            if node is not None:
#                if node.order_num != 0:
#                    reward_node_broadcast, finished_order_num_node_broadcast \
#                        = node.simple_order_assign_broadcast_update(self, neighbor_reward)
#                    reward += reward_node_broadcast
#                    #print "reward"
#                    #print reward
#                    finished_order_num += finished_order_num_node_broadcast

        node_reward = node_reward + neighbor_reward
        #print "node_reward"
        #print node_reward
        if all_order_num != 0:
            self.order_response_rate = finished_order_num / float(all_order_num)  # 这response_rate大于1？
        else:
            self.order_response_rate = -1

        return reward, [node_reward, neighbor_reward]  # 这里返回的时候，node_reward是包含了node_reward + neighbor_reward

    def step_remove_unfinished_orders(self):
        for node in self.nodes:
            if node is not None:
                node.remove_unfinished_order(self.city_time)  # 从node的orders中移除在此刻完成的订单和没有被服务且超过了等待时间的订单

    def step_pre_order_assigin(self, next_state):

        remain_drivers = next_state[0] - next_state[1]  # 司机减去订单的数量
        remain_drivers[remain_drivers < 0] = 0  # 如果订单数量大于司机数量 则司机数量设置为0

        remain_orders = next_state[1] - next_state[0]  # 同上 主体换为订单
        remain_orders[remain_orders < 0] = 0

        if np.sum(remain_orders) == 0 or np.sum(remain_drivers) == 0:  # 如果正好整个地图剩余的空闲司机或者订单数量为0
            context = np.array([remain_drivers, remain_orders])  # 直接返回context 不再进行以下操作
            return context

        remain_orders_1d = remain_orders.flatten()  # 剩余空闲司机矩阵2d->1d
        remain_drivers_1d = remain_drivers.flatten()  # 剩余待分配订单矩阵2d->1d
        # 一个遍历，将node的order分配解决，同时更新矩阵
        for node in self.nodes:
            if node is not None:  # 有效的node
                curr_node_id = node.get_node_index()  # 当前node的id
                if remain_orders_1d[curr_node_id] != 0:  # 如果剩余的订单不为0
                    for neighbor_node in node.neighbors:  # 遍历该节点的邻居节点，
                        if neighbor_node is not None:  # 邻居节点有效
                            neighbor_id = neighbor_node.get_node_index()  # 邻居节点的id
                            a = remain_orders_1d[curr_node_id]  # 邻居节点的订单数量
                            b = remain_drivers_1d[neighbor_id]  # 邻居节点的空闲司机数量
                            remain_orders_1d[curr_node_id] = max(a - b,
                                                                 0)  # 这里为什么不加一个判定if(remain_drivers_1d[neighbor_id]!=0),而是直接计算，这两种代码风格哪一种更好呢
                            remain_drivers_1d[neighbor_id] = max(b - a,
                                                                 0)  # 这里是一个greedy策略，因为他遍历的时候，根据遍历的顺序，在前面的node总是可以优先索取neighbor节点的空闲司机
                        if remain_orders_1d[curr_node_id] == 0:  # 当前节点的order得到满足，break
                            break

        context = np.array([remain_drivers_1d.reshape(self.M, self.N),
                            remain_orders_1d.reshape(self.M, self.N)])
        return context

    def step_dispatch_invalid(self, dispatch_actions):
        """ If a
        :param dispatch_actions:
        :return:
        """
        #print "dispatch_actions"
        #print dispatch_actions
        save_remove_id = []
        for action in dispatch_actions:

            start_node_id, end_node_id, num_of_drivers = action
            if self.nodes[start_node_id] is None or num_of_drivers == 0:
                continue  # not a feasible action

            if self.nodes[start_node_id].get_driver_numbers() < num_of_drivers:
                num_of_drivers = self.nodes[start_node_id].get_driver_numbers()

            if end_node_id < 0:
                for _ in np.arange(num_of_drivers):
                    self.nodes[start_node_id].set_idle_driver_offline_random()  # 随机让一个闲置的driver下线
                    self.n_drivers -= 1
                    self.n_offline_drivers += 1
                    self.all_grids_off_number += 1
                continue

            if self.nodes[end_node_id] is None:
                for _ in np.arange(num_of_drivers):
                    self.nodes[start_node_id].set_idle_driver_offline_random()
                    self.n_drivers -= 1
                    self.n_offline_drivers += 1
                    self.all_grids_off_number += 1
                continue
            # print self.nodes[start_node_id].get_node_index
            # print self.nodes[start_node_id].neighbors
            # print self.nodes[end_node_id]
            if self.nodes[end_node_id] not in self.nodes[start_node_id].neighbors:  # 如果终点不是出发点的邻节点
                raise ValueError('City:step(): not a feasible dispatch')

            for _ in np.arange(num_of_drivers):  # 前面排除错误后，开始调度
                # t = 1 dispatch start, idle driver decrease
                remove_driver_id = self.nodes[start_node_id].remove_idle_driver_random()  # 这是要调度的driver的id
                save_remove_id.append((end_node_id, remove_driver_id))  # 所以save_remove_id储存的事终点id和调度的driver的id的数组
                self.drivers[remove_driver_id].set_position(None)  # 说明driver在调度过程中的position一直都是none？
                self.drivers[remove_driver_id].set_offline_for_start_dispatch()  # 设置driver离线
                self.n_drivers -= 1

        return save_remove_id

    def step_add_dispatched_drivers(self, save_remove_id):
        # drivers dispatched at t, arrived at t + 1
        for destination_node_id, arrive_driver_id in save_remove_id:
            self.drivers[arrive_driver_id].set_position(self.nodes[destination_node_id])
            self.drivers[arrive_driver_id].set_online_for_finish_dispatch()
            self.nodes[destination_node_id].add_driver(arrive_driver_id, self.drivers[arrive_driver_id])
            self.n_drivers += 1

    def step_increase_city_time(self):
        self.city_time += 1
        # set city time of drivers
        for driver_id, driver in self.drivers.iteritems():
            driver.set_city_time(self.city_time)

    def step(self, reward_cost,dispatch_actions, generate_order=1):
        info = []
        reward_cost = reward_cost
        '''**************************** T = 1 ****************************'''
        # Loop over all dispatch action, change the driver distribution
        save_remove_id = self.step_dispatch_invalid(dispatch_actions)  # save_remove_id储存的是终点id和调度的driver的id的数组
        # When the drivers go to invalid grid, set them offline.

        reward, reward_node = self.step_assign_order_broadcast_neighbor_reward_update(reward_cost)  # reward_node=[node_reward, neighbor_reward]

        '''**************************** T = 2 ****************************'''
        # increase city time t + 1
        self.step_increase_city_time()
        self.step_driver_status_control()  # drivers finish order become available again.

        # drivers dispatched at t, arrived at t + 1, become available at t+1
        self.step_add_dispatched_drivers(
            save_remove_id)  # 这个函数是给调度的司机设计，功能是，在t时刻受调度的司机，在t+1时刻到达了目的地，然后set司机position为目的地node，设置司机状态为online,目的地num_driver+1

        # generate order at t + 1
        if generate_order == 1:
            self.step_generate_order_real()
        else:
            moment = self.city_time % self.n_intervals
            self.step_bootstrap_order_real(self.day_orders[moment])

        # offline online control;
        self.step_driver_online_offline_nodewise()  # 根据输入的onoff_driver_location_mat ，对司机的数量进行更新，也就是每个时间段 对司机数量真实变化的模拟

        self.step_remove_unfinished_orders()  # 移除在此刻完成的订单和没有被服务且超过了等待时间的订单
        # get states S_{t+1}  [driver_dist, order_dist]
        next_state = self.get_observation()

        context = self.step_pre_order_assigin(next_state)
        # print "context"
        # print context
        info = [reward_node, context]  # 这里的info和一开始初始化的时候不一样了好像
        #print reward_node
        #print "info"
        #print info
        return next_state, reward, info
