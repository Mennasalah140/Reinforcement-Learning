import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np

class ActorCriticModel(nn.Module):
    """Shared Network for Policy Gradient methods (A2C, PPO)."""
    def __init__(self, state_dim, action_dim, is_discrete):
        super(ActorCriticModel, self).__init__()
        self.is_discrete = is_discrete
        
        # --- Shared Feature Layers ---
        self.common = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # --- Actor (Policy) Head ---
        if self.is_discrete:
            self.actor_head = nn.Linear(128, action_dim)
        else:
            self.actor_mean = nn.Linear(128, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim)) 
        
        # --- Critic (Value) Head ---
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x):
        features = self.common(x)
        value = self.critic_head(features)
        
        if self.is_discrete:
            action_logits = self.actor_head(features)
            return action_logits, value
        else:
            mean = torch.tanh(self.actor_mean(features)) 
            log_std = self.actor_log_std.expand_as(mean)
            std = torch.exp(log_std)
            return mean, std, value