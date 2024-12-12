import torch
import os
import numpy as np
import numpy.random as rd
import pandas as pd
import pyomo.environ as pyo
import pyomo.kernel as pmo
from omlt import OmltBlock

from gurobipy import *
from omlt.neuralnet import NetworkDefinition, FullSpaceNNFormulation, ReluBigMFormulation
from omlt.io.onnx import write_onnx_model_with_bounds, load_onnx_neural_network_with_bounds
import tempfile
import torch.onnx
import torch.nn as nn
from copy import deepcopy
from random_generator_battery import ESSEnv



class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, gpu_id=0):
        self.now_len = 0
        self.next_idx = 0  # 下一个样本应该存储的位置索引
        self.if_full = False
        self.max_len = max_len
        self.data_type = torch.float32
        self.action_dim = action_dim
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        other_dim = 1 + 1 + self.action_dim  # r + done + action
        self.buf_other = torch.empty(size=(max_len, other_dim), dtype=self.data_type, device=self.device)

        if isinstance(state_dim, int):  # state is pixel
            self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)
        elif isinstance(state_dim, tuple):
            self.buf_state = torch.empty((max_len, *state_dim), dtype=torch.uint8, device=self.device)
        else:
            raise ValueError('state_dim')

    def extend_buffer(self, state, other):  # CPU array to CPU array
        # other = (reward, mask, action) mask = (1 - done) * gamma
        size = len(other)  # size是要加入的元组(state,(reward, mask, action))个数
        next_idx = self.next_idx + size

        if next_idx > self.max_len:
            self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
            self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True

            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

    def sample_batch(self, batch_size) -> tuple:
        indices = rd.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],  # r_m_a[:, 0:1]能够转化为一个（n,1）二维向量，r_m_a[:, 0]能够转化为一个（n,）一维向量
                r_m_a[:, 1:2],
                r_m_a[:, 2:],
                self.buf_state[indices],
                self.buf_state[indices + 1])  # return (reward,mask,action,state,next_state)

    def update_now_len(self):
        self.now_len = self.max_len if self.if_full else self.next_idx


class Actor(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim))

    def forward(self, state):
        return self.net(state).tanh()  # tanh() make the data from -1 to 1

    def get_action(self, state, action_std):
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)  # clamp(-0.5,0.5)将noise限制在(-0.5,0.5)之间
        return (action + noise).clamp(-1.0, 1.0)


class CriticQ(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_head = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, mid_dim), nn.ReLU())
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, 1))  # we get q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, 1))  # we get q2 value

    def forward(self, value):
        mid = self.net_head(value)
        return self.net_q1(mid)

    def get_q1_q2(self, value):
        mid = self.net_head(value)
        return self.net_q1(mid), self.net_q2(mid)


class AgentBase:
    def __init__(self):
        self.state = None
        self.device = None
        self.action_dim = None
        self.if_off_policy = None
        self.explore_noise = None
        self.trajectory_list = None
        self.explore_rate = 1.0

        self.criterion = torch.nn.SmoothL1Loss()

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, _if_per_or_gae=False, gpu_id=0):
        self.device = torch.device(
            f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.action_dim = action_dim

        self.cri = self.ClassCri(net_dim, state_dim, action_dim).to(self.device)
        self.act = self.ClassAct(net_dim, state_dim, action_dim).to(
            self.device) if self.ClassAct else self.cri
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
        self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate)
        del self.ClassCri, self.ClassAct

    def select_action(self, state) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)  # 先将state升维
        action = self.act(states)[0]
        if rd.rand() < self.explore_rate:
            action = (action + torch.randn_like(action) * self.explore_noise).clamp(-1, 1)
        return action.detach().cpu().numpy()  # 转化为numpy类型，因为需要与env交互

    def explore_env(self, env, target_step):
        trajectory = list()

        state = self.state
        for _ in range(target_step):
            action = self.select_action(state)

            state, next_state, reward, done, = env.step(action)

            trajectory.append((state, (reward, done, *action)))
            state = env.reset() if done else next_state
        self.state = state

        return trajectory

    @staticmethod  # 静态方法不需要 self 参数，这意味着它们不能直接访问类的实例属性或方法.
    # 尽管静态方法不接收 self 参数，但它们仍然是类的一部分，可以通过类名直接调用
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd, if_save):
        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)

        name_obj_list = [('actor', self.act), ('act_target', self.act_target), ('act_optim', self.act_optim),
                         ('critic', self.cri), ('cri_target', self.cri_target), ('cri_optim', self.cri_optim), ]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]
        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None

    def _update_exploration_rate(self, explorate_decay, explore_rate_min):
        self.explore_rate = max(self.explore_rate * explorate_decay, explore_rate_min)
        '''this function is used to update the explorate probability when select action'''


class AgentMIPDQN(AgentBase):
    def __init__(self):
        super().__init__()
        self.explore_noise = 0.5  # standard deviation of exploration noise
        self.policy_noise = 0.2  # standard deviation of policy noise
        self.update_freq = 2  # delay update frequency
        self.if_use_cri_target = self.if_use_act_target = True
        self.ClassCri = CriticQ
        self.ClassAct = Actor

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        buffer.update_now_len()
        obj_critic = obj_actor = None
        for update_c in range(int(buffer.now_len / batch_size * repeat_times)):  # we update too much time?
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic)

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri_target(
                torch.cat((state, action_pg), dim=-1)).mean()  # use cri_target instead of cri for stable training
            self.optim_update(self.act_optim, obj_actor)
            if update_c % self.update_freq == 0:  # delay update
                self.soft_update(self.cri_target, self.cri, soft_update_tau)
                self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_critic.item() / 2, obj_actor.item()

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise,
            next_q = torch.min(*self.cri_target.get_q1_q2(torch.cat((next_s, next_a), dim=-1)))  # twin critics
            q_label = reward + mask * next_q

        q1, q2 = self.cri.get_q1_q2(torch.cat((state, action), dim=-1))
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)  # twin critics
        return obj_critic, state


def update_buffer(buffer, _trajectory):
    # trajectory:=(state, (reward, done, *action))
    ten_state = torch.as_tensor([item[0] for item in _trajectory], dtype=torch.float32)  # 能够将tuple类型全部转化为torch类型
    ary_other = torch.as_tensor([item[1] for item in _trajectory])

    ary_other[:, 0] = ary_other[:, 0]  # ten_reward
    ary_other[:, 1] = (1.0 - ary_other[:, 1]) * gamma  # ten_mask = (1.0 - ary_done) * gamma
    ary_other[:, 2] = ary_other[:, 2]  # ten_action
    buffer.extend_buffer(ten_state, ary_other)

    _steps = ten_state.shape[0]
    _r_exp = ary_other[:, 0].mean()  # other = (reward, mask, *action)
    return _steps, _r_exp


def get_episode_return(env, act, device):
    '''get information of one episode during the training'''
    episode_return = 0.0  # sum of rewards in an episode
    episode_unbalance = 0.0
    episode_operation_cost = 0.0
    state = env.reset()
    for i in range(24):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        state, next_state, reward, done, = env.step(action)
        state = next_state
        episode_return += reward
        episode_unbalance += env.real_unbalance
        episode_operation_cost += env.operation_cost
        if done:
            break
    return episode_return, episode_unbalance, episode_operation_cost

class Actor_MIP:
    '''this actor is used to get the best action and Q function, the only input should be batch tensor state, action, and network, while the output should be
    batch tensor max_action, batch tensor max_Q'''
    def __init__(self,scaled_parameters,batch_size,net,state_dim,action_dim,env,constrain_on=False):
        self.batch_size = batch_size
        self.net = net
        self.state_dim = state_dim
        self.action_dim =action_dim
        self.env = env
        self.constrain_on=constrain_on
        self.scaled_parameters=scaled_parameters

    def get_input_bounds(self,input_batch_state):
        batch_size = self.batch_size
        batch_input_bounds = []
        lbs_states = input_batch_state.detach().numpy()
        ubs_states = lbs_states

        for i in range(batch_size):
            input_bounds = {}
            for j in range(self.action_dim + self.state_dim):
                if j < self.state_dim:
                    input_bounds[j] = (float(lbs_states[i][j]), float(ubs_states[i][j]))
                else:
                    input_bounds[j] = (float(-1), float(1))
            batch_input_bounds.append(input_bounds)
        return batch_input_bounds

    def predict_best_action(self, state):
        state=state.detach().cpu().numpy()
        v1 = torch.zeros((1, self.state_dim+self.action_dim), dtype=torch.float32)
        '''this function is used to get the best action based on current net'''
        model = self.net.to('cpu')
        input_bounds = {}
        lb_state = state
        ub_state = state
        for i in range(self.action_dim + self.state_dim):
            if i < self.state_dim:
                input_bounds[i] = (float(lb_state[0][i]), float(ub_state[0][i]))
            else:
                input_bounds[i] = (float(-1), float(1))

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            # export neural network to ONNX
            torch.onnx.export(
                model,
                v1,
                f,
                input_names=['state_action'],
                output_names=['Q_value'],
                dynamic_axes={
                    'state_action': {0: 'batch_size'},
                    'Q_value': {0: 'batch_size'}
                }
            )
            # write ONNX model and its bounds using OMLT
        write_onnx_model_with_bounds(f.name, None, input_bounds)
        # load the network definition from the ONNX model
        network_definition = load_onnx_neural_network_with_bounds(f.name)
        # global optimality
        formulation = ReluBigMFormulation(network_definition)
        m = pyo.ConcreteModel()
        m.nn = OmltBlock()
        m.nn.build_formulation(formulation)
        '''# we are now building the surrogate model between action and state'''
        # constrain for battery，
        if self.constrain_on:
            m.power_balance_con1 = pyo.Constraint(expr=(
                    (-m.nn.inputs[7] * self.scaled_parameters[0])+\
                    ((m.nn.inputs[8] * self.scaled_parameters[1])+m.nn.inputs[4]*self.scaled_parameters[5]) +\
                    ((m.nn.inputs[9] * self.scaled_parameters[2])+m.nn.inputs[5]*self.scaled_parameters[6]) +\
                    ((m.nn.inputs[10] * self.scaled_parameters[3])+m.nn.inputs[6]*self.scaled_parameters[7])>=\
                    m.nn.inputs[3] *self.scaled_parameters[4]-self.env.grid.exchange_ability))
            m.power_balance_con2 = pyo.Constraint(expr=(
                    (-m.nn.inputs[7] * self.scaled_parameters[0])+\
                    (m.nn.inputs[8] * self.scaled_parameters[1]+m.nn.inputs[4]*self.scaled_parameters[5]) +\
                    (m.nn.inputs[9] * self.scaled_parameters[2]+m.nn.inputs[5]*self.scaled_parameters[6]) +\
                    (m.nn.inputs[10] * self.scaled_parameters[3]+m.nn.inputs[6]*self.scaled_parameters[7])<=\
                    m.nn.inputs[3] *self.scaled_parameters[4]+self.env.grid.exchange_ability))
        m.obj = pyo.Objective(expr=(m.nn.outputs[0]), sense=pyo.maximize)

        pyo.SolverFactory('gurobi').solve(m, tee=False)

        best_input = pyo.value(m.nn.inputs[:])

        best_action = (best_input[self.state_dim::])
        return best_action

if __name__ == '__main__':

    # initialize
    '''here record real unbalance'''
    reward_record = {'episode': [], 'steps': [], 'mean_episode_reward': [], 'unbalance': [],
                     'episode_operation_cost': []}
    loss_record = {'episode': [], 'steps': [], 'critic_loss': [], 'actor_loss': [], 'entropy_loss': []}
    num_episode = 10
    gamma = 0.995  # discount factor of future rewards
    learning_rate = 1e-4  # 2 ** -14 ~= 6e-5
    soft_update_tau = 1e-2  # 2 ** -8 ~= 5e-3

    net_dim = 64  # the network width 256
    batch_size = 256  # num of transitions sampled from replay buffer.
    repeat_times = 2 ** 3  # repeatedly update network to keep critic's loss small
    target_step = 1000  # collect target_step experiences , then update network, 1024
    max_memo = 50000  # capacity of replay buffer
    ## arguments for controlling exploration
    explorate_decay = 0.99
    explorate_min = 0.3
    if_per_or_gae = None

    # initialize agent
    agent = AgentMIPDQN()
    agent.if_use_cri_target = True
    agent.if_use_act_target = True
    env = ESSEnv()
    agent.state = env.reset()
    agent.init(net_dim, env.state_space.shape[0], env.action_space.shape[0], learning_rate,
               if_per_or_gae)
    '''init replay buffer'''
    buffer = ReplayBuffer(max_len=max_memo, state_dim=env.state_space.shape[0],
                          action_dim=env.action_space.shape[0])

    collect_data = True
    while collect_data:
        print(f'buffer:{buffer.now_len}')
        with torch.no_grad():
            trajectory = agent.explore_env(env, target_step)
            steps, r_exp = update_buffer(buffer, trajectory)
            buffer.update_now_len()
        if buffer.now_len >= 10000:
            collect_data = False
    for i_episode in range(num_episode):
        critic_loss, actor_loss = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)
        loss_record['critic_loss'].append(critic_loss)
        loss_record['actor_loss'].append(actor_loss)
        with torch.no_grad():
            episode_reward, episode_unbalance, episode_operation_cost = get_episode_return(env, agent.act,
                                                                                           agent.device)
            reward_record['mean_episode_reward'].append(episode_reward)
            reward_record['unbalance'].append(episode_unbalance)
            reward_record['episode_operation_cost'].append(episode_operation_cost)
        print(
            f'curren epsiode is {i_episode}, reward:{episode_reward},unbalance:{episode_unbalance},buffer_length: {buffer.now_len}')
        if i_episode % 10 == 0:
            # target_step
            with torch.no_grad():
                agent._update_exploration_rate(explorate_decay, explorate_min)
                trajectory = agent.explore_env(env, target_step)
                steps, r_exp = update_buffer(buffer, trajectory)

    print(loss_record)
