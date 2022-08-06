import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Net(nn.Module):
    def __init__(self, act_num, sentence_size, embedding_dim, filter_num=256, kernel_list=(3, 4, 5), dropout=0.75):
        super(Net, self).__init__()

        self.act_num = act_num
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, filter_num, (kernel, embedding_dim)),
                          nn.LeakyReLU(),
                          nn.MaxPool2d((sentence_size - kernel + 1, 1)))
            for kernel in kernel_list
        ])
        self.fc = nn.Linear(filter_num * len(kernel_list), 1024)
        self.dropout = nn.Dropout(dropout)


        # action policy layers
        self.act_fc1 = nn.Linear(1024, 2048)
        self.act_fc2 = nn.Linear(2048, 512)
        self.act_fc3 = nn.Linear(512, act_num)

        # state value layers
        self.val_fc1 = nn.Linear(1024, 2048)
        self.val_fc2 = nn.Linear(2048, 512)
        self.val_fc3 = nn.Linear(512, 1)

    def forward(self, state_input):
        state_input = state_input.unsqueeze(1)
        x = [conv(state_input) for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        # action policy layers
        x_act = F.relu(self.act_fc1(x))
        x_act = F.relu(self.act_fc2(x_act))
        x_act = self.act_fc3(x_act)
        x_act = F.log_softmax(x_act, dim=-1)

        # state value layers
        x_val = F.relu(self.val_fc1(x))
        x_val = F.relu(self.val_fc2(x_val))
        x_val = self.val_fc3(x_val)
        x_val = torch.tanh(x_val)

        return x_act, x_val

class PolicyValueNet():
    def __init__(self, act_num, sentence_size, embedding_dim, filter_num=256, kernel_list=(3, 4, 5), dropout=0.8,
                 model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.l2_const = 1e-4
        if self.use_gpu:
            self.policy_value_net = Net(act_num, sentence_size, embedding_dim,
                                        filter_num=filter_num, kernel_list=kernel_list, dropout=dropout).cuda()
        else:
            self.policy_value_net = Net(act_num, sentence_size, embedding_dim,
                                        filter_num=filter_num, kernel_list=kernel_list, dropout=dropout)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)

    def load_model(self, model_file):
        net_params = torch.load(model_file)
        self.policy_value_net.load_state_dict(net_params)

    def policy_value_fn(self, state):
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor([state]).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return list(act_probs[0]), value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor([state]))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return list(act_probs[0]), value.data.numpy()

    def policy_value(self, state_batch):
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def train_step(self, state_batch, probs, value_batch, lr):
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            probs = Variable(torch.FloatTensor(probs).cuda())
            value_batch = Variable(torch.FloatTensor(value_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            probs = Variable(torch.FloatTensor(probs))
            value_batch = Variable(torch.FloatTensor(value_batch))
        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)


        log_act_probs, value = self.policy_value_net(state_batch)
        value_loss = F.mse_loss(value.view(-1), value_batch)
        policy_loss = -torch.mean(torch.sum(probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        entropy = -torch.mean(
            torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
        )

        return loss.item(), value_loss.item(), policy_loss.item(), entropy.item()


    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        net_params = self.get_policy_param()
        torch.save(net_params, model_file)
