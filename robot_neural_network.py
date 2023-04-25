import tensorflow as tf
import numpy as np

from collections import namedtuple, deque
import random
import math
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Normal

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))


""" =========== Deep Q-Network =========== """
class ReplayMemoryDQN(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):
    """
    Deep Q-Network for Optimal Policy

    """
    def __init__(self, n_observations, n_actions, layer_size):
        super(DQN, self).__init__()

        # Set layer sizes
        self.layer1 = nn.Linear(n_observations, layer_size[0])
        self.layer2 = nn.Linear(layer_size[0], layer_size[1])
        self.layer3 = nn.Linear(layer_size[1], layer_size[2])
        self.output_layer = nn.Linear(layer_size[2], n_actions)
        
        # Set number of actions for use later

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.output_layer(x)
       

def select_action_DQN(obs, n_actions, policy_net, device, epsilon):
    # boolean to select random exploration with a probability of `epsilon`, else choose greedy action
    explore = True if random.random() < epsilon else False

    if explore: # select a random action
        action = random.choice([i for i in range(n_actions)])
        return torch.tensor([[action]], device = device, dtype = torch.long)
    else:
        with torch.no_grad():
            action = policy_net(obs).max(1)[1].view(1, 1)
            return action
    
def update_weights(policy_net, target_net, optimizer, memory, batch_size, gamma, device):
    """
    Method to optimise the DQN
    """
    if len(memory) < batch_size:
        return
    
    # sample data from the memory based on batch size
    transitions_memory = memory.sample(batch_size)
    batch_data = Transition(*zip(*transitions_memory ))

    non_final_next_stages_mask = torch.tensor(tuple([i != None for i in batch_data.next_state]), device = device)
    non_final_next_stages = torch.cat([i for i in batch_data.next_state if i != None])

    # get the state, action and rewards from the batch
    state_batch = torch.cat(batch_data.state)
    action_batch = torch.cat(batch_data.action)
    reward_batch = torch.cat(batch_data.reward)

    q_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(batch_size, device=device)

    # Calculating expected q values
    with torch.no_grad():
        next_state_values[non_final_next_stages_mask] = target_net(non_final_next_stages).max(1)[0]
    
    expected_q_values = (next_state_values * gamma) + reward_batch

    # Calculating loss
    criteria = nn.SmoothL1Loss()
    loss = criteria(q_values, expected_q_values.unsqueeze(1))
    # Calculating the backpropagation
    optimizer.zero_grad()
    loss.backward()
    
    # clip the neural network weight values
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss


""" =========== Soft-Actor Critic =========== """
# Implementation inspired from https://github.com/pranz24/pytorch-soft-actor-critic

def update(target, source, tau):
    """Update Method for SAC Only

    Args:
        target: Target Parameters to change
        source: Origin Parameters
        tau (float): Soft Update Policy Value
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class QNetwork(nn.Module):
    """Q-Network For SAC Model
    """
    def __init__(self, num_inputs, num_actions, hidden_layer_size):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.linear3 = nn.Linear(hidden_layer_size, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_layer_size)
        self.linear5 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.linear6 = nn.Linear(hidden_layer_size, 1)

        self.apply(init_weights_)

    def forward(self, state, action):
        # forward pass for SAC Q Networks
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2
class ReplayMemorySAC:
    """ReplayMemory for SAC
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

def init_weights_(m):
    """Initialise model weights"""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
    
class GaussianPolicy(nn.Module):
    """
    Generates Gaussian Policy whereby Outputs follow N(0,1)
    """
    def __init__(self, num_inputs, num_actions, hidden_layer_size, epsilon, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.epsilon = epsilon

        self.linear1 = nn.Linear(num_inputs, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, hidden_layer_size)

        self.mean_linear = nn.Linear(hidden_layer_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_layer_size, num_actions)

        self.apply(init_weights_)

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)

        else:
            action_space_high = 6.15
            action_space_low = -6.15
            self.action_scale = torch.tensor((action_space_high - action_space_low) / 2., dtype=torch.float64) 
            self.action_bias = torch.tensor((action_space_high + action_space_low) / 2., dtype=torch.float64)
    
    def forward(self, state):
        """forward pass
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        """Selects action based on prevailing output of Policy Network

        Args:
            state (np.array): observations of state

        Returns:
            action (int): action of index
            log_prob (tf.float): log of probability
            mean (tf.float): mean of action_probs
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        """Sets the action and bias to device

        Args:
            device (str): Device Type

        Returns:
            None: Sets Model to Device Type
        """
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
    
class SAC(object):
    """
    Soft Actor-Critic
    """
    def __init__(self, num_inputs, update_interval, hidden_layer_size, learning_rate, device, epsilon, gamma, tau, alpha = 0.2):
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.update_interval = update_interval

        self.device = device
        # self.hidden_layer_size = hidden_layer_size
        self.hidden_layer_size = 128 #192
        # self.learning_rate = learning_rate
        self.learning_rate = 0.0003
        action_space = 1

        self.critic = QNetwork(num_inputs = num_inputs, num_actions = action_space, hidden_layer_size = self.hidden_layer_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.learning_rate)

        self.critic_target = QNetwork(num_inputs = num_inputs, num_actions = action_space, hidden_layer_size = self.hidden_layer_size).to(device=self.device)
        update(target = self.critic_target, source = self.critic, tau = 1)

        self.policy = GaussianPolicy(num_inputs = num_inputs, num_actions = action_space, hidden_layer_size = self.hidden_layer_size, action_space = action_space, epsilon = self.epsilon).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr = self.learning_rate)

    def select_action(self, state, epsilon, evaluate = False):
        """
        Selects action based on epsilon greedy algorithm

        Args:
            state (np.array): array of observations
            epsilon (float): epsilon-greedy value 
            evaluate (bool, optional): False for training, True for testing. Defaults to False.

        Returns:
            action (int): action
        """
        import random
        explore = True if random.random() < epsilon else False

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        if explore: #random action
            action = random.uniform(-6.15, 6.15) # max velocity of wheels
            action = torch.tensor([[action]], device=self.device, dtype=torch.float)

        else: # greedy
            if evaluate == False:
                action, _, _ = self.policy.sample(state)
            else:
                _, _, action = self.policy.sample(state)
        
        return action.detach().cpu().numpy()[0]
    
    def update_weights(self, memory, batch_size, updates):
        """update model weights
        """
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if updates % self.update_interval == 0:
            update(self.critic_target, self.critic, self.tau)

        return policy_loss.detach()
    
    def save_checkpoint(self, env_name, suffix="", checkpoint_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if checkpoint_path is None:
            checkpoint_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(checkpoint_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, evaluate=False):
        print('Loading models from {}'.format(checkpoint_path))
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()


""" =========== Actor Critic & REINFORCE =========== """
class FullyConnectedLayersBlock(tf.keras.layers.Layer):
    """
    Constructor Class for the custom layers in a fully connected NN
    """
    def __init__(self, hidden_layer_size, weight_decay, dropout_rate):
        super(FullyConnectedLayersBlock,self).__init__()

        print(f"h_units: {hidden_layer_size}")
        self.dense = tf.keras.layers.Dense(hidden_layer_size, use_bias = False, 
                                           kernel_regularizer = tf.keras.regularizers.l2(l = weight_decay))
        
        self.batch_normalization = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training = False):
        x = self.dense(inputs)
        x = tf.nn.relu(x)
        x = self.batch_normalization(x, training = training)
        return x

        return 

class FullyConnectedModel(tf.keras.Model):    
    def __init__(self, model, hidden_layer_size, weight_decay, dropout_rate, num_of_outputs):
        super(FullyConnectedModel, self).__init__()
        self.model_name = None
        self.model = model
        self.checkpoint_dir = "Saved_Models/best_models/"
        self.checkpoint_path = None

        self.blocks = [FullyConnectedLayersBlock(hidden_layer_size[i], weight_decay[i], dropout_rate[i]) for i in range(3)]
        
        # Sets Output Layers for A2C actor and Reinforce;
        # Uses softmax for the action-probability 
        if self.model == "MAA2C_Actor" or self.model == 'Reinforce':
            self.outputs = tf.keras.layers.Dense(num_of_outputs, activation = 'softmax')

        # Sets Output Layers for A2C critic;
        # Outputs a float value for critic value of State 
        elif self.model == "MAA2C_Critic":
            self.outputs = tf.keras.layers.Dense(num_of_outputs)

        #For A2C, Combines the Actor and Critic Values into a single output
        elif self.model == "A2C":
            self.outputs_critic = tf.keras.layers.Dense(1)
            self.outputs_actions = tf.keras.layers.Dense(num_of_outputs, activation = 'softmax')

    def call(self, inputs, training =False):
        for i in range(3):
            x = self.blocks[i](inputs, training = training)
            inputs = x

        if self.model == "A2C":
            state_value = self.outputs_critic(x)
            probability_actions = self.outputs_actions(x)
            probability_actions_list = [probability_actions]
            return state_value, probability_actions_list
        
        else:
            x = self.outputs(x)
            return x

