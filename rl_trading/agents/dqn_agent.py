# agents/dqn_agent.py

import numpy as np
import random
from collections import deque
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.optim as optim

# ─────────────────────────────────────────────
# 1. Q-NETWORK  (the "brain")
# ─────────────────────────────────────────────
class QNetwork(nn.Module):
    """
    Maps state → Q-value for every action.

    Input : state vector  (obs_dim,)
    Output: Q-values      (action_dim,)  e.g. [Q(Hold), Q(Buy), Q(Sell)]

    Architecture: 3-layer MLP with ReLU.
    Simple is better here — financial data is noisy,
    a deep net would overfit.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# 2. REPLAY BUFFER  (the "memory")
# ─────────────────────────────────────────────
class ReplayBuffer:
    """
    Stores (state, action, reward, next_state, done) tuples.

    WHY: Without this, we'd train on consecutive timesteps which
    are highly correlated (today's price ≈ yesterday's price).
    Random sampling breaks this correlation — same reason we
    shuffle batches in supervised ML.

    capacity=10_000 means we keep the last 10k transitions.
    """
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        # Unzip into separate arrays — much faster than looping
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
# 3. DQN AGENT  (puts it all together)
# ─────────────────────────────────────────────
class DQNAgent:
    """
    Deep Q-Network agent for equity trading.

    Key hyperparameters explained:
        gamma       : discount factor. 0.95 = tuned for intraday day-trading.
                      Agent plans ~20 candles (~5 hours) ahead meaningfully,
                      then discounts heavily — incentivises closing positions
                      within the same session rather than holding overnight.
                      (0.95^25 ≈ 0.28 vs 0.99^25 ≈ 0.78 at same horizon)
        epsilon     : exploration rate. Starts at 1.0 (fully random),
                      decays toward epsilon_min as agent learns.
        lr          : learning rate for Adam optimizer.
        batch_size  : number of transitions sampled per training step.
        target_update_freq : how often to copy online → target network.
    """

    def __init__(
        self,
        obs_dim:             int,
        action_dim:          int   = 3,
        hidden_dim:          int   = 128,
        gamma:               float = 0.95,   # intraday-tuned (was 0.99)
        lr:                  float = 1e-3,
        epsilon:             float = 1.0,
        epsilon_min:         float = 0.01,
        epsilon_decay:       float = 0.995,
        batch_size:          int   = 64,
        buffer_capacity:     int   = 10_000,
        target_update_freq:  int   = 100,
    ):
        self.action_dim          = action_dim
        self.gamma               = gamma
        self.epsilon             = epsilon
        self.epsilon_min         = epsilon_min
        self.epsilon_decay       = epsilon_decay
        self.batch_size          = batch_size
        self.target_update_freq  = target_update_freq
        self.learn_step_counter  = 0   # tracks when to sync target net

        # Use MPS (Apple Silicon) if available, else CPU
        self.device = (
            torch.device("mps")  if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        print(f"DQN using device: {self.device}")

        # Online network — trained every step
        self.online_net = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)

        # Target network — frozen copy, synced every target_update_freq steps
        # WHY: If we compute targets with the same network we're updating,
        # the target moves every step → training becomes unstable (chasing a
        # moving target). A frozen copy gives stable supervision.
        self.target_net = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # target net never needs gradients

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()
        self.buffer    = ReplayBuffer(buffer_capacity)

    # ── Action selection ─────────────────────────────────────
    def select_action(self, state: np.ndarray) -> int:
        """
        ε-greedy policy:
          - With probability ε  → random action  (EXPLORE)
          - With probability 1-ε → best Q action (EXPLOIT)

        Early in training ε≈1 so agent explores broadly.
        As ε decays, agent increasingly trusts its learned Q-values.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax().item())

    # ── Store transition ─────────────────────────────────────
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    # ── Learn from a batch ───────────────────────────────────
    def learn(self):
        """
        One gradient update step using the Bellman equation.

        Bellman target:
            Q_target(s,a) = r  +  γ · max_a' Q_target(s', a')
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                   best future Q from TARGET net

        Loss:
            MSE( Q_online(s,a) ,  Q_target(s,a) )

        We only update the online net — target net is frozen.
        """
        # Don't learn until buffer has enough samples
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Convert to tensors
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Q(s, a) — current online net predictions for taken actions
        q_pred = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Q_target(s', a') — best Q from frozen target net
        with torch.no_grad():
            q_next   = self.target_net(next_states_t).max(1)[0]
            q_target = rewards_t + self.gamma * q_next * (1 - dones_t)

        loss = self.loss_fn(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — prevents exploding gradients (common in finance)
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Sync target network periodically
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    # ── Save / Load ──────────────────────────────────────────
    def save(self, path: str):
        torch.save({
            'online_net':  self.online_net.state_dict(),
            'target_net':  self.target_net.state_dict(),
            'optimizer':   self.optimizer.state_dict(),
            'epsilon':     self.epsilon,
        }, path)
        print(f"Model saved → {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded ← {path}")