import torch
import torch.nn as nn
import torch.optim as optim
from pg_agent_base import PolicyGradientAgentBase
from torch.distributions import Categorical, Normal
import numpy as np
from actor_critic_model import ActorCriticModel


class PPOAgent(PolicyGradientAgentBase):
    def __init__(self, state_dim, action_dim, device, hyperparams, is_discrete):
        super().__init__(state_dim, action_dim, device, hyperparams, is_discrete)
        
        self.CLIP_EPSILON = hyperparams.get('clip_epsilon', 0.2)
        self.PPO_EPOCHS = hyperparams.get('ppo_epochs', 10)
        self.MINIBATCH_SIZE = hyperparams.get('minibatch_size', 64)
        self.ENTROPY_COEFF = hyperparams.get('entropy_coeff', 0.01)
        self.old_model = ActorCriticModel(state_dim, action_dim, is_discrete).to(device)
        self.old_model.load_state_dict(self.model.state_dict())
        
    def get_log_prob_and_value(self, model, state, action):
        """Helper to get policy log probability, entropy, and value from a model."""
        if self.is_discrete:
            logits, value = model(state)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)
            entropy = dist.entropy().mean()
        else:
            mean, std, value = model(state)
            dist = Normal(mean, std)
            # Log probability of the sampled action
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().mean() 
            
        return log_prob, value, entropy
    
    def learn(self, next_state_value):
        """Calculates Discounted Returns, GAE (Advantage), and performs PPO updates."""
        if not self.rewards:
            return None

        # Store the current model state into the 'old' model for PPO ratio calculation
        self.old_model.load_state_dict(self.model.state_dict())

        # Compute Returns and Advantage ---
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        values = torch.cat(self.values).detach() 
        
        # Calculate Discounted Returns (R_t) backwards
        R = next_state_value.detach()
        returns = []
        for reward, mask in zip(reversed(self.rewards), reversed(self.masks)):
            R = reward.to(self.device) + self.GAMMA * R * mask.to(self.device)
            returns.append(R)
        returns = torch.cat(list(reversed(returns)))

        # Advantage: A_t = R_t - V(s)
        advantage = returns - values
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6) 

        # Get old log probabilities for the ratio calculation
        with torch.no_grad():
            old_log_probs, _, _ = self.get_log_prob_and_value(self.old_model, states, actions)
        
        # PPO Optimization Loop 
        data_size = len(states)
        all_indices = np.arange(data_size)
        total_loss = 0
        
        for _ in range(self.PPO_EPOCHS):
            np.random.shuffle(all_indices)
            
            for start in range(0, data_size, self.MINIBATCH_SIZE):
                end = start + self.MINIBATCH_SIZE
                batch_indices = all_indices[start:end]

                # Sample batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantage = advantage[batch_indices]
                
                # Get current log_prob and value from the policy network
                log_prob, value, entropy = self.get_log_prob_and_value(self.model, batch_states, batch_actions)
                
                # CRITIC LOSS (Value Function)
                critic_loss = 0.5 * (value - batch_returns).pow(2).mean()

                # ACTOR LOSS (Clipped Policy Gradient)
                # Policy Ratio: r(θ) = π_new / π_old
                ratio = torch.exp(log_prob - batch_old_log_probs)
                
                # Clipped Policy Gradient Objective
                surr1 = ratio * batch_advantage
                surr2 = torch.clamp(ratio, 1.0 - self.CLIP_EPSILON, 1.0 + self.CLIP_EPSILON) * batch_advantage
                actor_loss = -torch.min(surr1, surr2).mean()

                # FINAL LOSS (PPO Objective)
                loss = actor_loss + critic_loss - self.ENTROPY_COEFF * entropy
                total_loss += loss.item()
                
                # OPTIMIZE
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Clear trajectory buffers (On-Policy)
        self.clear_trajectory_buffers()
        
        return total_loss / self.PPO_EPOCHS