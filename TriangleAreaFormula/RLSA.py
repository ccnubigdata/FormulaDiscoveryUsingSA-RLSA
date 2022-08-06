import numpy as np
import random
import copy
import datetime
from sympy import *
from TriangleAreaFormula.fitting import linear_fitting, deal_result
from TriangleAreaFormula.policy_value_net import PolicyValueNet



Ra, Rb, Rc, Ba, Bb, Bc, Ma, Mb, Mc, A, B, C, a, b, c, s, r, R, Ha, Hb, Hc, S, Empty, Empty1 \
    = symbols('Ra Rb Rc Ba Bb Bc Ma Mb Mc A B C a b c s r R Ha Hb Hc Area $ #')
rotate_setting = [
    {c: Empty, b: c, a: b, Empty: a},
    {Hc: Empty, Hb: Hc, Ha: Hb, Empty: Ha},
    {Rc: Empty, Rb: Rc, Ra: Rb, Empty: Ra},
    {Bc: Empty, Bb: Bc, Ba: Bb, Empty: Ba},
    {Mc: Empty, Mb: Mc, Ma: Mb, Empty: Ma},
    {C: Empty, B: C, A: B, Empty: A},
]


def normalization(_pro, available_postion):
    for idx, f in enumerate(available_postion):
        _pro[idx] *= f
    _sum = sum(_pro)
    pro = np.zeros_like(_pro)
    for idx, value in enumerate(_pro):
        pro[idx] = value / _sum
    return pro

def get_rotate(ori_exp, l):
    rotate_list = [ori_exp]
    number = 0
    for times in range(2):
        new_exp = rotate_list[number]
        for regulation in rotate_setting:
            for key in regulation:
                new_exp = new_exp.subs({key: regulation[key]})
        if new_exp in l:
            number += 1
            rotate_list.append(new_exp)
    return list(set(rotate_list))

def load_var(path, name_var, name_data):
    ch_ori_var = np.load(path + name_var + '.npy', allow_pickle=True)
    data = np.load(path + name_data + '.npy', allow_pickle=True).item()
    ori_var = []
    for v in ch_ori_var:
        f = True
        if v == 0:
            f = False
        else:
            v_data = data[v]
            for d in v_data:
                if d*d < 0:
                    f = False
                    break
        if f is True:
            ori_var.append(v)

    vis = list()
    var = list()

    ori_var = sorted(ori_var, key=lambda x: len(str(x)))
    for it in ori_var:
        if it not in vis:
            ch = get_rotate(it, ori_var)
            vis.extend(ch)
            for i in range(len(ch), 3):
                ch.append(Empty)
            var.append(ch)

    data.update({Empty : np.zeros_like(data[var[0][0]])})
    return var, data


class Agent():
    def __init__(self, CODE_SIZE, CODE_X_SIZE, CODE_Y_SIZE, ACT_NUM_X, ACT_NUM_Y, MAX_T, MIN_T, RATE_T, MEMORY_SIZE,
                 batch_size, result_path=None):
        self.CODE_SIZE = CODE_SIZE
        self.CODE_X_SIZE = CODE_X_SIZE
        self.CODE_Y_SIZE = CODE_Y_SIZE
        self.ACT_NUM_X = ACT_NUM_X
        self.ACT_NUM_Y = ACT_NUM_Y
        self.MAX_T = MAX_T
        self.MIN_T = MIN_T
        self.RATE_T = RATE_T
        self.act_num = self.ACT_NUM_X * self.CODE_X_SIZE + self.ACT_NUM_Y * self.CODE_Y_SIZE
        self.memory = []
        self.memory_index = 0
        self.MEMORY_SIZE = MEMORY_SIZE
        self.flag_learn = True
        self.start_learn = False
        self.batch_size = batch_size
        self.epochs = 3  # num of train_steps for each update
        self.kl_targ = 0.02
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0

        if result_path is None:
            self.result = {}
        else:
            self.result = np.load(result_path, allow_pickle=True).item()

    def load_data(self, path, name_data_x, name_var_x, name_data_y, name_var_y, model_file):

        self.var_x, self.data_x = load_var(path, name_var_x, name_data_x)
        self.var_y, self.data_y = load_var(path, name_var_y, name_data_y)

        self.X_SIZE = len(self.var_x)
        self.Y_SIZE = len(self.var_y)
        self.policy_value_net = PolicyValueNet(act_num=self.act_num, sentence_size=self.CODE_SIZE*3,
                                               embedding_dim=len(self.data_x[self.var_x[0][0]]), use_gpu=True, model_file=model_file)
        self.pre_policy_value_net = PolicyValueNet(act_num=self.act_num, sentence_size=self.CODE_SIZE*3,
                                                embedding_dim=len(self.data_x[self.var_x[0][0]]), use_gpu=True)

        self.idx_to_varx = {i: j[0] for i, j in enumerate(self.var_x)}
        self.varx_to_idx = {j[0]: i for i, j in enumerate(self.var_x)}

        self.idx_to_vary = {i: j[0] for i, j in enumerate(self.var_y)}
        self.vary_to_idx = {j[0]: i for i, j in enumerate(self.var_y)}

        # 创建邻接矩阵
        self.adj_x = np.zeros([self.X_SIZE, self.ACT_NUM_X])
        ave_value = list()
        len_value = len(self.data_x[self.var_x[0][0]])
        for var in self.var_x:
            sum_value = 0
            len_var = 0
            for it_var in var:
                if it_var is not None:
                    len_var += 1
                    sum_value += sum(self.data_x[it_var])
            ave_value.append([var, sum_value / (len_value * len_var)])
        ave_value = sorted(ave_value, key=lambda x: x[1])
        for cur in ave_value:
            sub_ave = list()
            for x in ave_value:
                if x is not cur:
                    sub_ave.append([x[0], abs(cur[1] - x[1])])
            sub_ave = sorted(sub_ave, key=lambda x: x[1])
            l, r = 0, len(sub_ave) - 1
            m = (l + r) // self.ACT_NUM_X
            for i in range(self.ACT_NUM_X):
                self.adj_x[self.varx_to_idx[cur[0][0]]][i] = int(self.varx_to_idx[sub_ave[i * m][0][0]])


        self.adj_y = np.zeros([self.Y_SIZE, self.ACT_NUM_Y])
        ave_value = list()
        len_value = len(self.data_y[self.var_y[0][0]])
        for var in self.var_y:
            sum_value = 0
            len_var = 0
            for it_var in var:
                if it_var is not None:
                    len_var += 1
                    sum_value += sum(self.data_y[it_var])
            ave_value.append([var, sum_value / (len_value * len_var)])
        ave_value = sorted(ave_value, key=lambda x: x[1])
        for cur in ave_value:
            sub_ave = list()
            for x in ave_value:
                if x is not cur:
                    sub_ave.append([x[0], abs(cur[1] - x[1])])
            sub_ave = sorted(sub_ave, key=lambda x: x[1])
            l, r = 0, len(sub_ave) - 1
            m = (l + r) // self.ACT_NUM_Y
            for i in range(self.ACT_NUM_Y):
                self.adj_y[self.vary_to_idx[cur[0][0]]][i] = int(self.vary_to_idx[sub_ave[i * m][0][0]])

    def get_state(self, code=None):
        if code is None:
            code = self.code
        state = []
        for i in range(self.CODE_SIZE):
            idx = code[i]
            if i < self.CODE_X_SIZE:
                var = self.var_x[idx]
                for v in var:
                    state.append(self.data_x[v])
            else:
                var = self.var_y[idx]
                for v in var:
                    state.append(self.data_y[v])
        return state

    def init_state(self):
        self.step_num = 0
        self.code = random.sample(range(self.X_SIZE), self.CODE_X_SIZE)
        self.code.extend(random.sample(range(self.Y_SIZE), self.CODE_Y_SIZE))
        self.state = self.get_state()
        self.get_available_postion()
        self.pro, self.fitness= self.policy_value_net.policy_value_fn(self.state)
        self.pro = normalization(self.pro, self.available_postion)
        self.cur_T = self.MAX_T
        self.search_record = []

    def get_available_postion(self):
        self.available_postion = np.ones(self.CODE_X_SIZE * self.ACT_NUM_X +
                      self.CODE_Y_SIZE * self.ACT_NUM_Y)
        for idx in range(self.act_num):
            if idx < self.CODE_X_SIZE * self.ACT_NUM_X:
                num = idx // self.ACT_NUM_X
                act = idx % self.ACT_NUM_X
                if self.adj_x[self.code[num]][act] in self.code:
                    self.available_postion[idx] = 0
            else:
                num = (idx - self.CODE_X_SIZE * self.ACT_NUM_X) // self.ACT_NUM_Y + self.CODE_X_SIZE
                act = (idx - self.CODE_X_SIZE * self.ACT_NUM_X) % self.ACT_NUM_Y
                if self.adj_y[self.code[num]][act] in self.code:
                    self.available_postion[idx] = 0

    def select_neighbor(self, pro, neighbor_num=1):
        neighbor = np.random.choice(a=len(pro), size=neighbor_num, p=pro*0.7+0.3*np.random.dirichlet(np.ones(len(pro))))
        best_neighbor_fitness = -2
        neighbor_code = []
        for idx in neighbor:
            # 解码
            ch_neighbor_code = copy.deepcopy(self.code)
            if idx < self.CODE_X_SIZE * self.ACT_NUM_X:
                num = idx // self.ACT_NUM_X
                act = idx % self.ACT_NUM_X
                ch_neighbor_code[num] = int(self.adj_x[ch_neighbor_code[num]][act])
            else:
                num = (idx - self.CODE_X_SIZE * self.ACT_NUM_X) // self.ACT_NUM_Y + self.CODE_X_SIZE
                act = (idx - self.CODE_X_SIZE * self.ACT_NUM_X) % self.ACT_NUM_Y
                ch_neighbor_code[num] = int(self.adj_y[ch_neighbor_code[num]][act])
            if self.flag_learn is not True:
                ch_fitness = np.tanh(self.calculate_error(code=ch_neighbor_code))
            else:
                _, ch_fitness = self.policy_value_net.policy_value_fn(self.get_state(ch_neighbor_code))
            if ch_fitness > best_neighbor_fitness:
                best_neighbor_fitness = ch_fitness
                neighbor_code = copy.deepcopy(ch_neighbor_code)
        return neighbor_code, best_neighbor_fitness

    def code_to_data(self, code=None):
        if code is None:
            code = self.code
        data_x, data_y = [], []
        for i in range(self.CODE_SIZE-1):
            idx = code[i]
            if i < self.CODE_X_SIZE:
                var = self.var_x[idx]
                for v in var:
                    if v is not Empty:
                        data_x.append(self.data_x[v])
            else:
                var = self.var_y[idx]
                for v in var:
                    if v is not Empty:
                        data_x.append(self.data_y[v])
        var = self.var_y[self.code[self.CODE_SIZE-1]]
        data = []
        for v in var:
            if v is not Empty:
                data.append(self.data_y[v])
        data_x.extend(data[:-1])
        data_y= [data[-1]]
        return data_x, data_y

    def calculate_error(self, code=None, reciprocal=True):
        if code is None:
            code = self.code
        x_data, y_data = self.code_to_data(code)
        final_score, arr, b = linear_fitting(x_data, y_data)
        if reciprocal is True:
            return 1. / float(final_score)
        else:
            return float(final_score)

    def get_simple_neighbor_pro(self):
        activate_act_num = sum(self.available_postion)
        pro = np.ones(self.CODE_X_SIZE * self.ACT_NUM_X +
                      self.CODE_Y_SIZE * self.ACT_NUM_Y)
        pro = pro / activate_act_num
        for idx, f in enumerate(self.available_postion):
            pro[idx] = f / activate_act_num

        return pro

    def isaccept(self, nfit):
        if nfit >= self.fitness:
            return True
        elif np.exp((float(nfit - self.fitness) / self.cur_T)) > random.random():
            return True
        else:
            return False

    def shown_result(self):
        x, y = [], []
        data_x, data_y = self.code_to_data()
        leny = 0
        for i in range(self.CODE_SIZE - 1):
            idx = self.code[i]
            if i < self.CODE_X_SIZE:
                var = self.var_x[idx]
                for v in var:
                    if v is not Empty:
                        x.append(v)
            else:
                var = self.var_y[idx]
                for v in var:
                    if v is not Empty:
                        x.append(v)
                        leny += 1
        var = self.var_y[self.code[self.CODE_SIZE - 1]]
        ch_var = []
        for v in var:
            if v is not Empty:
                ch_var.append(v)
                leny += 1
        leny -= 1
        x.extend(ch_var[:-1])
        y = [ch_var[-1]]
        return deal_result(x, y, data_x, data_y, leny)

    def is_endding(self):
        err = self.calculate_error(reciprocal=False)
        if err < 1e-8 or 1-np.abs(self.fitness[0]) < 1e-8:
            return True
        if self.cur_T > self.MIN_T:
            return False
        else:
            return True

    def real_reward(self, epoch):
        self.expression = self.shown_result()
        if self.expression == -1:
            return -1
        else:
            if self.expression not in self.result:
                self.result.update({epoch: self.expression})
            return 1

    def record_cur_state(self):
        value = np.zeros_like(self.pro)
        for idx, f in enumerate(self.available_postion):
            if f == 0:
                continue
            ch_code = copy.deepcopy(self.code)
            if idx < self.CODE_X_SIZE * self.ACT_NUM_X:
                num = idx // self.ACT_NUM_X
                act = idx % self.ACT_NUM_X
                ch_code[num] = int(self.adj_x[ch_code[num]][act])
            else:
                num = (idx - self.CODE_X_SIZE * self.ACT_NUM_X) // self.ACT_NUM_Y + self.CODE_X_SIZE
                act = (idx - self.CODE_X_SIZE * self.ACT_NUM_X) % self.ACT_NUM_Y
                ch_code[num] = int(self.adj_y[ch_code[num]][act])
            if self.flag_learn is not True:
                ch_fitness = np.exp(np.tanh(self.calculate_error(code=ch_code)))
            else:
                _, ch_fitness = self.pre_policy_value_net.policy_value_fn(self.get_state(ch_code))
                ch_fitness = np.exp(ch_fitness)
            value[idx] = ch_fitness
        _pro = np.zeros_like(self.pro)
        _sum = sum(value)
        for idx, v in enumerate(value):
            _pro[idx] = v / _sum + 1e-6
        self.search_record.append([self.state, _pro])

    def update(self, code):
        self.code = code
        self.state = self.get_state()
        self.get_available_postion()
        self.pro, self.fitness = self.policy_value_net.policy_value_fn(self.state)
        self.pro = normalization(self.pro, self.available_postion)

    def policy_update(self):
        self.pre_policy_value_net.policy_value_net.load_state_dict(self.policy_value_net.get_policy_param())
        mini_batch = random.sample(self.memory, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        probs_batch = [data[1] for data in mini_batch]
        succeed_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, value_loss, policy_loss, entropy = self.policy_value_net.train_step(
                state_batch,
                probs_batch,
                succeed_batch,
                self.learn_rate)
                # self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1)
                         )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        return loss, value_loss, policy_loss, entropy

    def save_result(self, path):
        np.save(path, self.result)

    def run(self, epoch, loss_, value_loss_, policy_loss_):
        self.init_state()
        count = 0
        while agnet.is_endding() is not True:
            neighbor_code, neighbor_fitness = self.select_neighbor(self.pro, 1)
            if self.isaccept(neighbor_fitness) is True:
                self.update(neighbor_code)
            self.cur_T *= self.RATE_T
            self.step_num += 1
        reward = self.real_reward(epoch)
        print(epoch+1, self.expression, self.code)

agnet = Agent(CODE_SIZE=5, CODE_X_SIZE=4, CODE_Y_SIZE=1, ACT_NUM_X=10, ACT_NUM_Y=10,
              MAX_T=5, MIN_T=1, RATE_T=0.995, MEMORY_SIZE=4000, batch_size=256)

agnet.load_data(path='datasets/', name_data_x='datavarx2_side_angle', name_var_x='varx2_side_angle',
                name_data_y='datavary2_side_angle', name_var_y='vary2_side_angle', model_file='model/side_angle_dim2.model')

t1 = datetime.datetime.now()
loss, value_loss, policy_loss = [], [], []
for i in range(500):
    agnet.run(i, loss, value_loss, policy_loss)
t2 = datetime.datetime.now()
print(t2-t1)
print(agnet.result)
agnet.save_result('result/EG_RLSA_result.npy')

