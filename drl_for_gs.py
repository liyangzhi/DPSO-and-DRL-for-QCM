import numpy as np
from env import Environment 
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import gym
import numpy as np
import argparse
 
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, batch_size):
        self.mem_size = max_size
        self.batch_size = batch_size
        self.mem_cnt = 0
 
        self.state_memory = np.zeros((self.mem_size, state_dim))
        self.action_memory = np.zeros((self.mem_size, ))
        self.reward_memory = np.zeros((self.mem_size, ))
        self.next_state_memory = np.zeros((self.mem_size, state_dim))
        self.terminal_memory = np.zeros((self.mem_size, ), dtype=np.bool_)
 
    def store_transition(self, state, action, reward, state_, done):
        mem_idx = self.mem_cnt % self.mem_size
 
        self.state_memory[mem_idx] = state
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.next_state_memory[mem_idx] = state_
        self.terminal_memory[mem_idx] = done
 
        self.mem_cnt += 1
 
    def sample_buffer(self):
        mem_len = min(self.mem_size, self.mem_cnt)
 
        batch = np.random.choice(mem_len, self.batch_size, replace=False)
 
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        terminals = self.terminal_memory[batch]
 
        return states, actions, rewards, states_, terminals
 
    def ready(self):
        return self.mem_cnt > self.batch_size
    

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
 
 
class DeepQNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(DeepQNetwork, self).__init__()
 
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, action_dim)
 
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)
 
    def forward(self, state):
        x = T.relu(self.fc1(state))
        x = T.relu(self.fc2(x))
 
        q = self.q(x)
 
        return q
 
    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)
 
    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))
 
 
class DDQN:
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim, ckpt_dir,
                 gamma=0.99, tau=0.005, epsilon=1.0, eps_end=0.01, eps_dec=5e-7,
                 max_size=1000000, batch_size=256):
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.checkpoint_dir = ckpt_dir
        self.action_space = [i for i in range(action_dim)]
 
        self.q_eval = DeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                   fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.q_target = DeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=fc1_dim, fc2_dim=fc2_dim)
 
        self.memory = ReplayBuffer(state_dim=state_dim, action_dim=action_dim,
                                   max_size=max_size, batch_size=batch_size)
 
        self.update_network_parameters(tau=1.0)
 
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
 
        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)
 
    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
 
    def choose_action(self, observation,acts, isTrain=True):
        state = T.tensor([observation], dtype=T.float).to(device)
        actions = self.q_eval.forward(state)
        #准备讲违法的动作赋予-inf
#         print(acts)
#         print(actions)
#         print(len(actions[0][0]))
        for i in range(len(actions[0][0])):
            #print('我运行')
            if i not in acts:
                actions[0][0][i]=-float('inf')#这里是不是写错了
        action = T.argmax(actions).item()
#         print(actions)
#         print(action)
#         assert 1==2
 
        if (np.random.random() < self.epsilon) and isTrain:
            action = np.random.choice(acts)
 
        return action
 
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min
 
    def learn(self):
        if not self.memory.ready():
            return
 
        states, actions, rewards, next_states, terminals = self.memory.sample_buffer()
        batch_idx = np.arange(self.batch_size)
 
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
        next_states_tensor = T.tensor(next_states, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)
 
        with T.no_grad():
            q_ = self.q_eval.forward(next_states_tensor)
            next_actions = T.argmax(q_, dim=-1)
            q_ = self.q_target.forward(next_states_tensor)
            q_[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * q_[batch_idx, next_actions]
        q = self.q_eval.forward(states_tensor)[batch_idx, actions]
 
        loss = F.mse_loss(q, target.detach())
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()
 
        self.update_network_parameters()
        self.decrement_epsilon()
 
    def save_models(self, episode):
        self.q_eval.save_checkpoint(self.checkpoint_dir + 'Q_eval/DDQN_q_eval_{}.pth'.format(episode))
        print('Saving Q_eval network successfully!')
        self.q_target.save_checkpoint(self.checkpoint_dir + 'Q_target/DDQN_Q_target_{}.pth'.format(episode))
        print('Saving Q_target network successfully!')
 
    def load_models(self, episode):
        self.q_eval.load_checkpoint(self.checkpoint_dir + 'Q_eval/DDQN_q_eval_{}.pth'.format(episode))
        print('Loading Q_eval network successfully!')
        self.q_target.load_checkpoint(self.checkpoint_dir + 'Q_target/DDQN_Q_target_{}.pth'.format(episode))
        print('Loading Q_target network successfully!')
        
        
    
 
 
def plot_learning_curve(episodes, records, title, ylabel, figure_file):
    plt.figure()
    plt.plot(episodes, records, linestyle='-', color='r')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)
 
    plt.show()
    plt.savefig(figure_file)
 
 
def create_directory(path: str, sub_dirs: list):
    for sub_dir in sub_dirs:
        if os.path.exists(path + sub_dir):
            print(path + sub_dir + ' is already exist!')
        else:
            os.makedirs(path + sub_dir, exist_ok=True)
            print(path + sub_dir + ' create successfully!')
            

max_episodes=500
#max_episodes=150
ckpt_dir='./checkpoints/DDQN/'
reward_path='./output_images/avg_reward.png'
epsilon_path='./output_images/epsilon.png'

#env = gym.make('LunarLander-v2')
env=Environment(ac,cs)
agent = DDQN(alpha=0.0003, state_dim=len(env.get_obs()), action_dim=env.action_space.n,
             fc1_dim=256, fc2_dim=256, ckpt_dir=ckpt_dir, gamma=0.99, tau=0.005, epsilon=1.0,
             eps_end=0.05, eps_dec=5e-6, max_size=100000, batch_size=256)
create_directory(ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
total_rewards, avg_rewards, eps_history = [], [], []
SWAPs=[]
maps=[]

for episode in range(max_episodes):
    total_reward = 0
    done = False
    istate=env.reset()
    observation =env.get_obs()
    #observation = np.array(istate).flatten()
    add_SWAP=0
    #print(observation)
    while not done:
        acts=env.get_act()
        action = agent.choose_action([observation],acts, isTrain=True)
        #assert 1==2
        add_SWAP+=1
        state_, reward, done, info = env.step(action)
        #observation_=np.array(observation_).flatten()
        #print(observation_)
        observation_=env.get_obs()
        agent.remember(observation, action, reward, observation_, done)#看是否需要加括号
        agent.learn()
        total_reward += reward
        observation = observation_

    total_rewards.append(total_reward)
    avg_reward = np.mean(total_rewards[-100:])
    avg_rewards.append(avg_reward)
    eps_history.append(agent.epsilon)
    print('EP:{} reward:{} avg_reward:{} epsilon:{}'.
          format(episode + 1, total_reward, avg_reward, agent.epsilon))
    print('SWAP count',add_SWAP)
    SWAPs.append(add_SWAP)
    maps.append(env.mp)
    
    

    if (episode + 1) % 50 == 0:
        agent.save_models(episode + 1)

episodes = [i for i in range(max_episodes)]
plot_learning_curve(episodes, avg_rewards, 'Reward', 'reward', reward_path)
plot_learning_curve(episodes, eps_history, 'Epsilon', 'epsilon', epsilon_path)