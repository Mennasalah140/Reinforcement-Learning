import torch
import torch.nn as nn
from pg_agent_base import PolicyGradientAgentBase
from torch.distributions import Categorical, Normal
import torch.optim as optim
from actor_critic_model import ActorCriticModel
import numpy as np

class PPOAgent(PolicyGradientAgentBase):
    def __init__(self, state_dim, action_dim, device, hyperparams, is_discrete):
        super().__init__(state_dim, action_dim, device, hyperparams, is_discrete)
        
        # PPO-specific hyperparameters
        self.CLIP_EPSILON = hyperparams.get('clip_epsilon', 0.2)
        self.PPO_EPOCHS = hyperparams.get('ppo_epochs', 10)
        self.MINIBATCH_SIZE = hyperparams.get('minibatch_size', 64)
        self.entropy_coeff = hyperparams.get('entropy_coeff', 0.02)

        # Old policy network for ratio calculation
        self.old_model = ActorCriticModel(state_dim, action_dim, is_discrete).to(device)
        self.old_model.load_state_dict(self.model.state_dict())

    def get_log_prob_and_value(self, state, action):
        """Helper to get policy log probability and value from the model."""
        if self.is_discrete:
            logits, value = self.model(state)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)
        else:
            mean, std, value = self.model(state)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return log_prob, value

    def learn(self, next_state_value):
        if not self.rewards:
            return None

        # --- 1. Combine trajectory data ---
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        old_log_probs = torch.cat(self.log_probs).detach()
        values = torch.cat(self.values)

        # --- 2. Compute discounted returns ---
        R = next_state_value.detach()
        returns = []
        for reward, mask in zip(reversed(self.rewards), reversed(self.masks)):
            R = reward.to(self.device) + self.GAMMA * R * mask.to(self.device)
            returns.append(R)
        returns = torch.cat(list(reversed(returns)))

        # --- 3. Compute advantages ---
        advantage = returns - values.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

        # --- 4. PPO update loop ---
        data_size = len(states)
        all_indices = np.arange(data_size)

        for _ in range(self.PPO_EPOCHS):
            np.random.shuffle(all_indices)

            for start in range(0, data_size, self.MINIBATCH_SIZE):
                end = start + self.MINIBATCH_SIZE
                batch_indices = all_indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantage = advantage[batch_indices]

                # Current log_prob and value
                log_prob, value = self.get_log_prob_and_value(batch_states, batch_actions)

                # --- Critic loss (value function) ---
                critic_loss = 0.5 * (value - batch_returns).pow(2).mean()

                # --- Actor loss (clipped surrogate) ---
                ratio = torch.exp(log_prob - batch_old_log_probs)
                surr1 = ratio * batch_advantage
                surr2 = torch.clamp(ratio, 1.0 - self.CLIP_EPSILON, 1.0 + self.CLIP_EPSILON) * batch_advantage
                actor_loss = -torch.min(surr1, surr2).mean()

                # --- Entropy bonus ---
                if self.is_discrete:
                    dist = Categorical(logits=self.model(batch_states)[0])
                else:
                    mean, std, _ = self.model(batch_states)
                    dist = Normal(mean, std)
                entropy = dist.entropy().mean()

                # --- Total loss ---
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coeff * entropy

                # --- Backpropagation ---
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # --- 5. Update old policy ---
        self.old_model.load_state_dict(self.model.state_dict())

        # --- 6. Clear trajectory buffers ---
        self.clear_trajectory_buffers()

        return loss.item()
