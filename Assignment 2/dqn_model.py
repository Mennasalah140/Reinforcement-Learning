import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) architecture.
    A simple Feedforward Neural Network used to estimate Q(s, a) values.
    """
    def __init__(self, n_observations, n_actions):
        """
        Initializes the network layers (similar to lecture slide example).
        Input: State (n_observations)
        Output: Q-value for each action (n_actions)
        """
        super(DQN, self).__init__()
        # Layers defined based on the example structure (e.g., [cite: 345])
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """Forward pass through the network."""
        # The input x might be a batch, hence the reshape if needed
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)