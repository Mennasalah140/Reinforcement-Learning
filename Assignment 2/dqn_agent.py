import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import numpy as np
from dqn_model import DQN
from replay_memory import ReplayMemory, Transition

class DQNAgent:
    def __init__(self, state_dim, action_dim, device, is_ddqn, hyperparams):
        # Store all hyperparams for easy access (especially seed for testing)
        self.hyperparams = hyperparams

        # Hyperparameters
        self.GAMMA = hyperparams['gamma']
        self.LR = hyperparams['learning_rate']
        self.MEMORY_CAPACITY = hyperparams['memory_size']
        self.BATCH_SIZE = hyperparams['batch_size']
        self.EPS_START = hyperparams['eps_start']
        self.EPS_END = hyperparams['eps_end']
        self.EPS_DECAY = hyperparams['eps_decay']
        
        self.is_ddqn = is_ddqn
        self.device = device
        self.action_dim = action_dim
        self.steps_done = 0

        # Set seeds for reproducibility
        seed = hyperparams.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Placeholder for continuous action space object (used by Pendulum)
        self.continuous_action_space = None 

        # Initialize Policy and Target Networks
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(self.MEMORY_CAPACITY)
        
    def select_action(self, state):
        """Epsilon-greedy action selection."""
        sample = random.random()
        
        # Calculate epsilon threshold (exponential decay)
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        
        if sample > eps_threshold:
            # Exploitation: Select best action from Policy Net
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # Exploration: Select a random action
            return torch.tensor([[random.randrange(self.action_dim)]], 
                                device=self.device, dtype=torch.long)

    def optimize_model(self):
        """Performs one step of optimization on the Policy Network."""
        if len(self.memory) < self.BATCH_SIZE:
            return None
        
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                        device=self.device, dtype=torch.bool)

        # Current Q-values: Q(s,a) for the actions taken
        Q_current = self.policy_net(state_batch).gather(1, action_batch)

        # Next state values initialization
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        
        # --- CORRECTED DDQN / DQN Logic ---
        with torch.no_grad():  # No gradients needed for target computation
            if self.is_ddqn:
                # DDQN: Policy Net selects action, Target Net evaluates it
                # 1. Use policy net to select best actions for next states
                next_actions = self.policy_net(non_final_next_states).argmax(1)
                # 2. Use target net to evaluate those actions
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # DQN: Target Net both selects and evaluates
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # Compute the Q-Target (R + gamma * V(s'))
        Q_target = reward_batch + (next_state_values * self.GAMMA)

        # Compute Smooth L1 Loss (Huber Loss)
        criterion = nn.SmoothL1Loss()
        loss = criterion(Q_current, Q_target.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability (critical for convergence)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        
        self.optimizer.step()
        
        return loss.item()

    def update_target_net(self):
        """Hard update: copies policy net weights to target net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())