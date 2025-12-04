import torch
import torch.nn as nn
import torch.optim as optim
from actor_critic_model import ActorCriticModel
from torch.distributions import Categorical, Normal
import numpy as np

class PolicyGradientAgentBase:
    def __init__(self, state_dim, action_dim, device, hyperparams, is_discrete):
        self.device = device
        self.hyperparams = hyperparams
        self.GAMMA = hyperparams['gamma']
        self.LR = hyperparams['learning_rate']
        self.is_discrete = is_discrete
        self.action_dim = action_dim

        # Trajectory Storage (On-Policy) 
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
        self.states = [] 
        self.actions = [] 
        
        # Network Initialization
        self.model = ActorCriticModel(state_dim, action_dim, is_discrete).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)
        self.action_scale = 2.0 

    def select_action(self, state):
        """Selects action based on policy distribution, handles continuous scaling."""
        state = state.to(self.device)
        
        if self.is_discrete:
            logits, value = self.model(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            
            self.log_probs.append(dist.log_prob(action).unsqueeze(-1))
            self.values.append(value)
            self.states.append(state)
            self.actions.append(action.unsqueeze(-1))
            
            return action.item()
        else:
            # Continuous Action Space (Pendulum)
            mean, std, value = self.model(state)
            dist = Normal(mean, std)
            action = dist.sample()
            
            # Policy is pi(a|s). Log probability of the sampled action.
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.states.append(state)
            self.actions.append(action) 
            env_action = action.cpu().numpy().flatten() * self.action_scale
            
            if env_action.ndim == 0:
                return np.array([env_action.item()], dtype=np.float32)
                
            return env_action
            
    def store_transition(self, reward, terminated):
        """Stores a transition for the current trajectory."""
        self.rewards.append(torch.tensor([[reward]], dtype=torch.float32))
        self.masks.append(torch.tensor([[1.0 - terminated]], dtype=torch.float32))

    def clear_trajectory_buffers(self):
        """Clears buffers after an update (necessary for On-Policy)."""
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
        self.states = []
        self.actions = []