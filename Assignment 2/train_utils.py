import math
import gymnasium as gym
import torch
import numpy as np
import wandb
import time
import random
from gymnasium.wrappers import RecordVideo

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_state(state, device):
    """Converts numpy state array to a PyTorch tensor."""
    if isinstance(state, tuple):
        state = state[0]
    return torch.from_numpy(state).float().unsqueeze(0).to(device)

def map_discrete_to_continuous(discrete_action_index, num_discrete_actions, continuous_space):
    """
    Maps a discrete action index (0, 1, 2, ...) back to a continuous torque value
    required by the Pendulum environment.
    """
    min_val = continuous_space.low[0]
    max_val = continuous_space.high[0]
    
    action_range = np.linspace(min_val, max_val, num_discrete_actions)
    
    # Return the corresponding continuous value
    return np.array([action_range[discrete_action_index]], dtype=np.float32)


def train_agent(env_name, agent, hyperparams, num_episodes, log_wandb=True):
    """
    Main training loop for a single agent on a single environment.
    Uses target_update_freq from hyperparams.
    """
    TARGET_UPDATE_FREQ = hyperparams.get('target_update_freq', 200)

    # Environment Setup
    env = gym.make(env_name)
    num_discrete_actions = 5 
    
    # PENDULUM : DISCRETIZE ACTION SPACE
    if 'Pendulum' in env_name:
        agent.continuous_action_space = env.action_space
        env.action_space = gym.spaces.Discrete(num_discrete_actions) 
    
    total_steps = 0
    
    for episode in range(num_episodes):
        # Reset environment
        state, info = env.reset()
        state = preprocess_state(state, DEVICE)
        
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            # 1. Action Selection
            action = agent.select_action(state)
            discrete_action_index = action.item()
            
            # 2. Step Environment with the CORRECT action type
            if 'Pendulum' in env_name:
                env_action = map_discrete_to_continuous(
                    discrete_action_index, 
                    num_discrete_actions, 
                    agent.continuous_action_space
                )
            else:
                env_action = discrete_action_index
            
            obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            
            reward = torch.tensor([reward], device=DEVICE)
            next_state = preprocess_state(obs, DEVICE) if not done else None
            
            # 3. Store Transition
            agent.memory.push(state, action, next_state, reward)
            
            state = next_state
            episode_reward += reward.item()
            episode_steps += 1
            total_steps += 1
            
            # 4. Optimize Model
            loss = agent.optimize_model()
            
            if log_wandb and loss is not None:
                wandb.log({"train/loss": loss}, step=total_steps)

            if total_steps % TARGET_UPDATE_FREQ == 0:
                agent.update_target_net()
            
        if log_wandb:
            # Log metrics at the end of the episode
            wandb.log({
                "train/episode_reward": episode_reward, 
                "train/episode_steps": episode_steps, 
                "train/epsilon": agent.EPS_END + (agent.EPS_START - agent.EPS_END) * math.exp(-1. * agent.steps_done / agent.EPS_DECAY)}, 
                step=total_steps
            )
            
        print(f"Episode {episode+1}/{num_episodes}, Steps: {episode_steps}, Reward: {episode_reward:.2f}")

    env.close()
    return agent

def test_agent(env_name, agent, num_tests=100, record_video=False):
    """
    Tests the trained agent for stability and records one video.
    """
    if record_video:
        video_dir = f"./videos/{env_name}_{'DDQN' if agent.is_ddqn else 'DQN'}"
        env = RecordVideo(gym.make(env_name, render_mode="rgb_array"), 
                            video_folder=video_dir, 
                            episode_trigger=lambda x: x == 0, 
                            name_prefix="Trained_Agent_Test")
    else:
        env = gym.make(env_name)
    
    num_discrete_actions = 5

    if 'Pendulum' in env_name:
        continuous_space = agent.continuous_action_space 
        env.action_space = gym.spaces.Discrete(num_discrete_actions) 
    
    test_durations = []
    rewards = []
    
    for test in range(num_tests):
        # Reset environment
        state, info = env.reset()
        state = preprocess_state(state, DEVICE)
        
        episode_steps = 0
        episode_reward = 0
        done = False
        
        while not done:
            # Use greedy policy for testing
            with torch.no_grad():
                discrete_action_index = agent.policy_net(state).max(1)[1].view(1, 1).item()
            
            if 'Pendulum' in env_name:
                env_action = map_discrete_to_continuous(
                    discrete_action_index, 
                    num_discrete_actions, 
                    continuous_space
                )
            else:
                env_action = discrete_action_index
            
            obs, reward, terminated, truncated, info = env.step(env_action)
            episode_reward += reward
            done = terminated or truncated
            
            state = preprocess_state(obs, DEVICE) if not done else None
            episode_steps += 1
            
            if episode_steps > 1000: 
                break 
                
        print(f"Test {test+1}/{num_tests}, Steps: {episode_steps}, Reward: {episode_reward:.2f}")
        test_durations.append(episode_steps)
        rewards.append(episode_reward)
        
    env.close()

    avg_duration = np.mean(test_durations)
    std_duration = np.std(test_durations)
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    return avg_duration, std_duration, test_durations, avg_reward, std_reward