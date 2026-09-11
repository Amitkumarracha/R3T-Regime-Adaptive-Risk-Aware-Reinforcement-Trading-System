"""
Double DQN Agent.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

from .network import VanillaQNetwork
from .replay_buffer import ReplayBuffer

class DoubleDQNAgent:
    def __init__(
        self, 
        obs_dim, 
        action_dim, 
        hidden_dim=128, 
        gamma=0.99, 
        lr=1e-3, 
        batch_size=64, 
        buffer_capacity=50000, 
        target_update_freq=500,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        seed=42
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps = 0
        self.seed = seed
        self.model_name = "DoubleDQN"
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_net = VanillaQNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = VanillaQNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.memory = ReplayBuffer(buffer_capacity)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        if not deterministic and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
            
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            return q_values.argmax(dim=1).item()
            
    def get_confidence(self, state: np.ndarray) -> float:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            probs = torch.softmax(q_values, dim=1)
            return probs.max().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.steps += 1
        
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        current_q = self.q_net(states).gather(1, actions)
        
        # Double DQN change: select action with q_net, evaluate with target_net
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1).unsqueeze(1)
            max_next_q = self.target_net(next_states).gather(1, next_actions)
            expected_q = rewards + (1 - dones) * self.gamma * max_next_q
            
        loss = self.loss_fn(current_q, expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.q_net.state_dict(), path)
        
    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())

