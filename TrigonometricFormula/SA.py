import numpy as np
from TrigonometricFormula.fitting import linear_fitting, deal_result
import random
import copy
import math

def fallten_list(l):
    if type(l) is not list or type(l[0]) is not list:
        return l
    nl = []
    for sl in l:
        nl.extend(fallten_list(sl))
    return nl

def CreateActionTable(size, action_num, adj):
    action_table = np.zeros([size, action_num]).astype(int)
    inv_action_table = []
    for idx in range(size):
        inv_action_table.append([])

    if action_num == 0:
        return action_table, inv_action_table

    # 求x动作表
    step = (size-1) // action_num

    for idx in range(size):
        idx_v = [[idx, -v] for idx, v in enumerate(adj[idx, :])]
        idx_v = sorted(idx_v, key=lambda x: x[1])

        for a_idx in range(action_num-1):
            action_table[idx, a_idx] = idx_v[(a_idx-1)*step][0]
            inv_action_table[idx_v[(a_idx-1)*step][0]].append(idx)

        action_table[idx, action_num-1] = (idx+1) % size
        inv_action_table[(idx+1) % size].append(idx)

    return action_table, inv_action_table

class SA(object):

    def __init__(self, var_x, data_x, len_code_x, adj, n_actions,
                 max_T=8, min_T=1, rate_T=0.995):
        self.var_x = var_x
        self.len_x = len(var_x)
        self.data_x = data_x
        self.len_code = len_code_x
        self.len_code_x = len_code_x

        self.n_neighbor = 5
        self.max_T = max_T
        self.min_T = min_T
        self.rate_T = rate_T

        self.left_state = []
        self.left_var = []

        self.n_actions = n_actions
        self.action_table, self.inv_action_table = CreateActionTable(self.len_x, self.n_actions, adj)
        self.times = 0

    def get_data(self, left):
        left_data = []
        for lvar in left:
            left_data.append(self.data_x[lvar])

        return left_data

    def count_fit(self, left_var=None):
        if left_var is None:
            left_var = copy.deepcopy(self.left_var)

        left = fallten_list(left_var)
        left_data = self.get_data(left)
        fit = linear_fitting(left_data[:-1], [left_data[-1]], epoch=30)

        return fit

    def get_expression(self):
        left = fallten_list(self.left_var)

        left_data = self.get_data(left)
        leny = 0

        right = [left[-1]]
        right_data = [left_data[-1]]

        left = left[:-1]
        left_data = left_data[:-1]

        return deal_result(left, right, left_data, right_data, leny)

    def find_neighbor(self):
        nei_left_state, nei_left_var = [], []
        nei_fit = float('inf')
        for i in range(self.n_neighbor):
            nleft_state = copy.deepcopy(self.left_state)
            nleft_var = copy.deepcopy(self.left_var)

            idx = random.randint(0, self.len_code_x - 1)
            select_idx = nleft_state[idx]

            ch_x = self.action_table[select_idx][random.randint(0, self.n_actions - 1)]
            while ch_x in self.left_state:
                ch_x = self.action_table[select_idx][random.randint(0, self.n_actions - 1)]

            nleft_state[idx] = ch_x
            nleft_var[idx] = self.var_x[ch_x]

            fit = self.count_fit(nleft_var)
            if fit < nei_fit:
                nei_left_state = nleft_state
                nei_left_var = nleft_var
                nei_fit = fit
        return nei_left_state, nei_left_var, nei_fit

    def isaccept(self, nerror):
        if self.cur_error >= nerror:
            return True
        elif np.exp((float(2*math.tanh(1/(nerror+1e-12)) - 2*math.tanh(1/(self.cur_error+1e-12)))
                    / self.cur_T)) > random.random():
            return True
        else:
            return False

    def init_state(self, model=0):
        self.left_state = random.sample(range(self.len_x), self.len_code_x)

        # self.left_state = [339, 290, 35]
        # self.right_state = [21]

        self.left_var = []

        for left_id in self.left_state:
            self.left_var.append(self.var_x[left_id])

        self.cur_error = self.count_fit()
        if model == 0:
            self.cur_T = self.max_T

    def update(self, nei_left_state, nei_left_var, nerror):
        self.left_state = copy.deepcopy(nei_left_state)
        self.left_var = copy.deepcopy(nei_left_var)
        self.cur_error = nerror

    def run(self, isshown=0):
        self.init_state()
        while self.cur_T > self.min_T:
            self.times += 1
            nei_left_state,  \
            nei_left_var, nerror = self.find_neighbor()
            if self.isaccept(nerror):
                self.update(nei_left_state,
                            nei_left_var, nerror)
            self.cur_T *= self.rate_T

            if isshown:
                if self.times % 30 == 0:
                    print('温度{}, 编码:{}'.format(
                        self.cur_T, [self.left_state]))
            if self.cur_error < 1e-10:
                exp = self.get_expression()
                if exp != -1:
                    times = self.times
                    self.times = 0
                    return exp, times, [self.left_state]
                else:
                    self.init_state(-1)
        print('No formula found...\n')
        return -1, -1, -1

