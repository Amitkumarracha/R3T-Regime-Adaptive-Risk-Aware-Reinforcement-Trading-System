"""
Full R3T Agent (Dueling Double DQN with confidence gating).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

from .network import DuelingQNetwork
from .replay_buffer import ReplayBuffer

class R3TAgent:
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
        seed=42,
        confidence_thresholds=None
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
        self.model_name = "R3TAgent"
        
        if confidence_thresholds is None:
            self.confidence_thresholds = {
                "low": 0.55,
                "medium": 0.65,
                "high": 0.80
            }
        else:
            self.confidence_thresholds = confidence_thresholds
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use DuelingQNetwork
        self.q_net = DuelingQNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DuelingQNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
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
            probs = torch.softmax(q_values, dim=1)
            
            raw_action = q_values.argmax(dim=1).item()
            confidence = probs.max().item()
            
            # Apply confidence gating on BUY actions (1, 2, 3, 4)
            if deterministic and raw_action in [1, 2, 3, 4]:
                if confidence < self.confidence_thresholds["low"]:
                    return 0 # Force HOLD
                elif confidence < self.confidence_thresholds["medium"]:
                    return min(raw_action, 1) # Max 25%
                elif confidence < self.confidence_thresholds["high"]:
                    return min(raw_action, 2) # Max 50%
                
            return raw_action
            
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
        
        # Double DQN style update
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

