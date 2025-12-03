import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

# A common constant to ensure log_std doesn't become too small
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianActor(nn.Module):
    """SAC Actor Network: Outputs mean and log_std, then applies reparameterization and tanh."""
    def __init__(self, state_dim, action_dim, log_std_min=LOG_STD_MIN, log_std_max=LOG_STD_MAX):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state, deterministic=False, with_logprob=True):
        net_out = self.net(state)
        mu = self.mu_layer(net_out)
        log_std = torch.tanh(self.log_std_layer(net_out))
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        std = torch.exp(log_std)

        # Reparameterization Trick [cite: 1241]
        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample() # Sample action
        
        # Compute log probability and apply Squashing (tanh)
        if with_logprob:
            log_prob = pi_distribution.log_prob(pi_action).sum(axis=-1, keepdim=True)
            # Correction term for tanh squashing
            log_prob -= torch.sum(torch.log(torch.clamp(1 - pi_action.pow(2), 1e-6, 1.0)), dim=-1, keepdim=True)
        else:
            log_prob = None

        # Apply tanh squashing to keep action within [-1, 1] bounds
        action = torch.tanh(pi_action)
        return action, log_prob, mu

class QFunction(nn.Module):
    """SAC Critic Network: Calculates Q(s, a)."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        # Concatenate state and action inputs
        sa = torch.cat([state, action], dim=-1)
        return self.net(sa)

class SacModel(nn.Module):
    """Container for SAC Actor and Double Critics."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = QFunction(state_dim, action_dim)
        self.q2 = QFunction(state_dim, action_dim)
        self.actor = SquashedGaussianActor(state_dim, action_dim)
        
    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)