import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    """
    A cyclic buffer to hold past experience tuples (Transitions).
    Used to store experiences and sample random batches for training.
    """
    def __init__(self, capacity):
        """Initialize memory with a fixed capacity."""
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        """Save a transition to memory."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a random batch of transitions for training."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Returns the current size of the memory."""
        return len(self.memory)