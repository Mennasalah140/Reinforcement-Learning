import math
import gymnasium as gym
import torch
import numpy as np
import wandb
import random 
from gymnasium.wrappers import RecordVideo
from pg_agent_base import PolicyGradientAgentBase 

class Transition(object):
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

class ReplayMemory(object):
    """Minimal ReplayMemory implementation for Off-Policy agents (SAC)."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        """Saves a transition (s, a, r, s', d)."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Samples a batch and converts Transition objects to tuples."""
        transitions = random.sample(self.memory, batch_size)
        
        return [(t.state, t.action, t.reward, t.next_state, t.done) for t in transitions]
    
    def __len__(self):
        return len(self.memory)
    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_state(state, device):
    """Converts numpy state array to a PyTorch tensor."""
    if isinstance(state, tuple):
        state = state[0]
    return torch.from_numpy(state).float().unsqueeze(0).to(device)

def map_discrete_to_continuous(discrete_action_index, num_discrete_actions, continuous_space):
    """Maps a discrete action index back to a continuous torque value (for Pendulum)."""
    min_val = continuous_space.low[0]
    max_val = continuous_space.high[0]
    action_range = np.linspace(min_val, max_val, num_discrete_actions)
    
    return np.array([action_range[discrete_action_index]], dtype=np.float32)


def train_agent_pg(env_name, agent, hyperparams, num_episodes, log_wandb=True):
    """
    Main training loop for Policy Gradient agents (A2C, PPO, SAC).
    """
    SEED = hyperparams.get('seed', None)
    
    # Environment Setup
    env = gym.make(env_name)
    num_discrete_actions = 5 
    is_discrete = hyperparams.get('discrete', True)

    if 'Pendulum' in env_name and is_discrete:
        agent.continuous_action_space = env.action_space
        env.action_space = gym.spaces.Discrete(num_discrete_actions) 
        
    total_steps = 0
    
    for episode in range(num_episodes):
        state, info = env.reset(seed=SEED if SEED is None else SEED + episode)
        state_np = state
        state = preprocess_state(state, DEVICE)
        
        episode_reward = 0
        episode_steps = 0
        loss = None 
        done = False
        
        while not done:
            # Action Selection: returns the ready-to-use action
            action_output = agent.select_action(state)
            
            # Prepare Environment Action
            env_action = action_output
            action_to_store = action_output

            # HANDLE DISCRETE ENVS (A2C/PPO on CartPole, Acrobot, MountainCar)
            if agent.__class__.__name__ in ['A2CAgent', 'PPOAgent', 'SACAgent'] and is_discrete:
                env_action = action_output 
            
            # HANDLE CONTINUOUS ENVS (A2C/PPO/SAC on Pendulum)
            if env.action_space.__class__.__name__ == 'Box':
                if not isinstance(env_action, np.ndarray) or env_action.shape == ():
                    env_action = np.array([env_action], dtype=np.float32)
                elif env_action.ndim == 0:
                    env_action = np.expand_dims(env_action, axis=0)

            obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            
            next_state_np = obs 
            next_state = preprocess_state(obs, DEVICE) if not done else None
            
            # Store Transition (Off-Policy vs On-Policy)
            if agent.__class__.__name__ == 'SACAgent':
                # SAC: Store the full transition into the ReplayMemory
                agent.store_transition(state_np, action_to_store, reward, next_state_np, terminated or truncated)
            else:
                # A2C/PPO (On-Policy): Store reward/mask into temporary trajectory buffers
                agent.store_transition(reward, terminated or truncated)
            
            state_np = next_state_np
            state = next_state
            
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            # SAC (Off-Policy) optimizes every step
            if agent.__class__.__name__ == 'SACAgent':
                loss = agent.learn()
                if log_wandb and loss is not None:
                    wandb.log({"train/loss": loss}, step=total_steps)


        # Optimization (On-Policy Update: A2C/PPO)
        if agent.__class__.__name__ in ['A2CAgent', 'PPOAgent'] and done:
            
            # Get the final V(s') estimate
            if state is None:
                next_state_value = torch.zeros(1, 1).to(DEVICE)
            else:
                with torch.no_grad():
                    # Get the Value estimate V(s') from the Critic head
                    if agent.is_discrete:
                        _, next_state_value = agent.model(state)
                    else: 
                        _, _, next_state_value = agent.model(state)
                        
            loss = agent.learn(next_state_value)
        
        # W&B Logging
        if log_wandb:
            wandb.log({
                "train/episode_reward": episode_reward, 
                "train/episode_steps": episode_steps,
                "train/total_steps": total_steps,
                "train/loss": loss if loss is not None else 0}, 
                step=total_steps
            )
            
        print(f"Episode {episode+1}/{num_episodes}, Steps: {episode_steps}, Reward: {episode_reward:.2f}, Loss: {loss:.4f}" if loss is not None else f"Episode {episode+1}/{num_episodes}, Steps: {episode_steps}, Reward: {episode_reward:.2f}")

    env.close()
    return agent

def test_agent(env_name, agent, num_tests=100, record_video=False):
    """
    Tests the trained agent for stability and records one video.
    Policy Gradient agents are greedy by selecting the max probability/mean action.
    """
    SEED = agent.hyperparams.get('seed', None)
    
    # --- Environment Setup ---
    if record_video:
        video_dir = f"videos/{env_name}_{agent.__class__.__name__.replace('Agent', '')}"
        env = RecordVideo(gym.make(env_name, render_mode="rgb_array"), 
                            video_folder=video_dir, 
                            episode_trigger=lambda x: x == 0, 
                            name_prefix="Trained_Agent_Test")
    else:
        env = gym.make(env_name)
    
    num_discrete_actions = 5
    
    # If a continuous environment was discretized for training
    if 'Pendulum' in env_name and hasattr(agent, 'is_discrete') and agent.is_discrete:
        continuous_space = agent.continuous_action_space 
        env.action_space = gym.spaces.Discrete(num_discrete_actions) 
    
    test_durations = []
    test_rewards = []
    
    for test in range(num_tests):
        seed_value = SEED if SEED is None else SEED + 1000 + test 
        state, info = env.reset(seed=seed_value)
        state = preprocess_state(state, DEVICE)
        
        episode_steps = 0
        done = False
        episode_reward = 0
        
        while not done:
            # Use greedy policy for testing
            with torch.no_grad():
                env_action = None 
                
                if agent.__class__.__name__ == 'SACAgent':
                    # SAC Policy: use deterministic mean action
                    action_output = agent.select_action(state, deterministic=True)
                    
                    if env.action_space.__class__.__name__ == 'Discrete':
                        # Map continuous output to discrete index for discrete test environments
                        action_value = action_output.item()
                        num_actions = env.action_space.n
                        bins = np.linspace(-1.0, 1.0, num_actions + 1)[1:-1]
                        discrete_action_index = np.digitize(action_value, bins)
                        env_action = int(discrete_action_index)
                    else:
                        env_action = action_output 

                elif agent.is_discrete:
                    # Discrete Policy (A2C/PPO): select the action with highest probability (argmax)
                    logits, _ = agent.model(state)
                    discrete_action_index = logits.argmax(dim=1).item()
                    
                    if 'Pendulum' in env_name:
                        env_action = map_discrete_to_continuous(discrete_action_index, num_discrete_actions, continuous_space)
                    else:
                        env_action = discrete_action_index
                        
                else:
                    # A2C/PPO Continuous Policy: use the mean output
                    mean, _, _ = agent.model(state)
                    env_action = mean.cpu().numpy().flatten()
            
            obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            
            state = preprocess_state(obs, DEVICE) if not done else None
            episode_steps += 1
            episode_reward += reward
            
            if episode_steps > 1000: break 

        test_durations.append(episode_steps)
        test_rewards.append(episode_reward)

    env.close()

    avg_duration = np.mean(test_durations)
    std_duration = np.std(test_durations)
    avg_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)
    
    return avg_duration, std_duration, test_durations, avg_reward, std_reward, test_rewards