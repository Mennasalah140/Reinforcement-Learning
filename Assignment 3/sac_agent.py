import torch
import torch.optim as optim
import torch.nn.functional as F
from sac_model import SacModel 
from train_utils_pg import ReplayMemory, Transition 
import numpy as np

class SACAgent:
    def __init__(self, state_dim, action_dim, device, hyperparams, is_discrete):
        self.device = device
        self.hyperparams = hyperparams 
        self.GAMMA = hyperparams['gamma']
        self.LR_ACTOR = hyperparams['learning_rate'] 
        self.LR_CRITIC = hyperparams.get('lr_critic', 1e-3)
        self.LR_ALPHA = hyperparams.get('lr_alpha', 1e-4)
        self.TAU = hyperparams.get('tau', 0.005) 
        self.BATCH_SIZE = hyperparams['batch_size']
        self.MEMORY_CAPACITY = hyperparams['memory_size']
        self.is_discrete = is_discrete
        
        self.action_dim = action_dim 
        self.action_scale = 2.0 

        # Entropy Regularization
        self.alpha = hyperparams.get('alpha_start', 0.2) 
        self.log_alpha = torch.tensor(np.log(self.alpha), dtype=torch.float32, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.LR_ALPHA)
        self.target_entropy = -torch.tensor(action_dim, dtype=torch.float32, device=device)

        # Network Initialization
        self.model = SacModel(state_dim, action_dim).to(device)
        self.target_model = SacModel(state_dim, action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.q_optimizer = optim.Adam(list(self.model.q1.parameters()) + list(self.model.q2.parameters()), lr=self.LR_CRITIC)
        self.actor_optimizer = optim.Adam(self.model.actor.parameters(), lr=self.LR_ACTOR)

        self.memory = ReplayMemory(self.MEMORY_CAPACITY) 

    def select_action(self, state, deterministic=False):
        """
        SAC uses stochastic policy for exploration (sampling) and scales the output.
        Returns the scaled action array for env.step().
        """
        with torch.no_grad():
            action, _, _ = self.model.actor(state, deterministic=deterministic, with_logprob=False)
            
            # Scale the action to the environment's actual bounds
            env_action = action.cpu().numpy().flatten() * self.action_scale
            
            if env_action.ndim == 0:
                return np.array([env_action.item()], dtype=np.float32)
                
            return env_action
    
    def store_transition(self, state, action, reward, next_state, terminated):
        """
        Stores a transition (s, a, r, s', done) in the Replay Memory.
        SAC needs to store the UN-SCALED, continuous action for learning purposes.
        """
        unscaled_action = action / self.action_scale
        
        state_np_flat = state.flatten()
        next_state_np_flat = next_state.flatten()
        action_np_flat = unscaled_action.flatten()
        
        self.memory.push(state_np_flat, action_np_flat, reward, next_state_np_flat, terminated)
    
    def learn(self):
        """Performs optimization on Critics, then Actor, then Alpha (entropy)."""
        if len(self.memory) < self.BATCH_SIZE: 
            return None 
        
        # 1. Sample Batch (Off-Policy)
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.as_tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        action_batch = torch.as_tensor(np.array(batch.action), dtype=torch.float32, device=self.device)
        reward_batch = torch.as_tensor(np.array(batch.reward), dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_state_batch = torch.as_tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        done_batch = torch.as_tensor(np.array(batch.done), dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        # 2. Critic Update
        with torch.no_grad():
            next_action, next_log_prob, _ = self.model.actor(next_state_batch)
            q1_target, q2_target = self.target_model(next_state_batch, next_action)
            min_q_target = torch.min(q1_target, q2_target)
            
            # Compute soft Bellman target (R + gamma * [min Q - alpha * H]) 
            next_v = min_q_target - self.alpha * next_log_prob
            q_target = reward_batch + self.GAMMA * (1 - done_batch) * next_v
        
        q1_current, q2_current = self.model(state_batch, action_batch)
        q1_loss = F.mse_loss(q1_current, q_target)
        q2_loss = F.mse_loss(q2_current, q_target)
        critic_loss = q1_loss + q2_loss
        
        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        # 3. Actor Update
        new_action, log_prob, _ = self.model.actor(state_batch)
        q1_new, q2_new = self.model(state_batch, new_action)
        min_q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_prob - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 4. Alpha Update (Entropy Tuning)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().item()
        
        # 5. Target Network Soft Update
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
            
        return critic_loss.item() + actor_loss.item()