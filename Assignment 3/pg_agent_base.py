import torch
import numpy as np
from actor_critic_model import ActorCriticModel
from torch.distributions import Categorical, Normal
import torch.optim as optim

class PolicyGradientAgentBase:
    """
    Base class for Policy Gradient Agents (A2C, PPO).
    Handles common functionality like action selection and trajectory storage.
    """
    def __init__(self, state_dim, action_dim, device, hyperparams, is_discrete):
        self.device = device
        self.hyperparams = hyperparams
        self.GAMMA = hyperparams['gamma']
        self.LR = hyperparams['learning_rate']
        self.is_discrete = is_discrete
        self.action_dim = action_dim

        # --- Trajectory Storage (On-Policy) ---
        self.states = [] 
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
        
        # --- Network Initialization ---
        self.model = ActorCriticModel(state_dim, action_dim, is_discrete).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)

        # Placeholder for continuous action space object (used by Pendulum)
        self.continuous_action_space = None 

    def select_action(self, state):
        """Selects action based on policy distribution."""
        state = state.to(self.device)
        
        # Store the current state tensor for the later batch update
        self.states.append(state)
        
        if self.is_discrete:
            # Discrete Action Space (CartPole, Acrobot, MountainCar)
            logits, value = self.model(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            
            # Store the action as a tensor (important for concatenation later)
            self.actions.append(action.unsqueeze(-1))
            self.log_probs.append(dist.log_prob(action).unsqueeze(-1))
            self.values.append(value)
            
            return action.item()
        else:
            # Continuous Action Space (Pendulum)
            mean, std, value = self.model(state)
            dist = Normal(mean, std)
            action = dist.sample()
            
            # Store the action as a tensor
            self.actions.append(action)
            # Store log_prob and value estimate
            self.log_probs.append(dist.log_prob(action).sum(dim=-1, keepdim=True)) 
            self.values.append(value)

            # Action needs to be scaled/clipped to the environment limits
            return action.cpu().numpy().flatten()


    def store_transition(self, reward, terminated):
        """Stores a transition for the current trajectory."""
        self.rewards.append(torch.tensor([[reward]], dtype=torch.float32))
        self.masks.append(torch.tensor([[1.0 - terminated]], dtype=torch.float32))

    def clear_trajectory_buffers(self):
        """Clears buffers after an update (necessary for On-Policy)."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []