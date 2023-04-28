import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class a2c(nn.Module):
    def __init__(self, 
                 obs_dim,
                 act_dim,
                 lr_actor=1e-4, 
                 lr_critic=1e-3, 
                 gamma=0.99,
                 device="cpu",
                 save_path=None,
                 tb_path=None):
        super(a2c, self).__init__()
        self.device = device
        self.gamma = gamma
        self.layer_size = 64
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, self.layer_size),
            nn.ReLU(),
            nn.Linear(self.layer_size,self.layer_size),
            nn.ReLU(),
            nn.Linear(self.layer_size, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, self.layer_size),
            nn.ReLU(),
            nn.Linear(self.layer_size,self.layer_size),
            nn.ReLU(),
            nn.Linear(self.layer_size, 1)
        )
        # self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        # self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.save_path = save_path
        self.tb_path = tb_path
        self.tb_writer = None

        if tb_path:
            self.tb_writer = SummaryWriter(log_dir=tb_path)

    def select_action(self, obs):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        obs_tensor =  obs_tensor.to(self.device)
        action_probs = self.actor(obs_tensor)

        probs = torch.softmax(action_probs, dim=1) # Hacky Way to do things, Must remember to normalise the input and output
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    def update(self, obs, action, reward, obs_next, done, log_prob,optimizer):

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        obs_tensor =  obs_tensor.to(self.device)
        obs_next_tensor = torch.FloatTensor(obs_next).unsqueeze(0)
        obs_next_tensor = obs_next_tensor.to(self.device)
        # print("obs_tensor Tensor device:", obs_next_tensor.device)

        critic_value = self.critic(obs_tensor)
        critic_value_next = self.critic(obs_next_tensor)
        delta = reward + self.gamma * critic_value_next * (1 - done) - critic_value
        actor_loss = -log_prob * delta.detach()
        critic_loss = delta.pow(2)

        self.loss= critic_loss + actor_loss
        optimizer.zero_grad()
        self.loss.backward()
        optimizer.step()

        # self.optimizer_actor.zero_grad()
        # actor_loss.backward()
        # self.optimizer_actor.step()
        # self.optimizer_critic.zero_grad()
        # critic_loss.backward()
        # self.optimizer_critic.step()

    def save(self, path=None):
        if not path:
            path = self.save_path
        if not path:
            raise ValueError("No save path specified")
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
            # 'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            # 'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
        }, path)

    def load(self, path=None):
        if not path:
            path = self.save_path
        if not path:
            raise ValueError("No save path specified")
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        # self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        # self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])

    def log(self, tag, value, step):
        if self.tb_writer:
            self.tb_writer.add_scalar(tag, value, step)

    