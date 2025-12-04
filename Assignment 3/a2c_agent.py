import torch
import torch.nn as nn
from pg_agent_base import PolicyGradientAgentBase

class A2CAgent(PolicyGradientAgentBase):
    def __init__(self, state_dim, action_dim, device, hyperparams, is_discrete):
        super().__init__(state_dim, action_dim, device, hyperparams, is_discrete)

    def learn(self, next_state_value):
        if not self.rewards:
            return None

        R = next_state_value.detach()
        returns = []
        for reward, mask in zip(reversed(self.rewards), reversed(self.masks)):
            R = reward.to(self.device) + self.GAMMA * R * mask.to(self.device)
            returns.append(R)
        returns = torch.cat(list(reversed(returns)))

        log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.values)
        
        advantage = returns - values.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)  

        # Critic loss
        critic_loss = 0.5 * advantage.pow(2).mean()

        # Actor loss + entropy
        entropy = -(log_probs * log_probs.exp()).sum(-1).mean()
        actor_loss = -(log_probs * advantage).mean() - 0.02 * entropy

        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.clear_trajectory_buffers()
        return loss.item()
