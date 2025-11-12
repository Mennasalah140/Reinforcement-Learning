import os
import torch
import wandb
import gymnasium as gym
from train_utils import train_agent, test_agent, DEVICE
from dqn_agent import DQNAgent

# Experiments
ENVIRONMENTS = [
    # 'CartPole-v1', 
    # 'Acrobot-v1', 
    # 'MountainCar-v0', 
    'Pendulum-v1' 
]

# HYPERPARAMETERS PER ENVIRONMENT
ENV_BASE_HYPERPARAMS = {
    'CartPole-v1': {
        'gamma': 0.99,
        'learning_rate': 0.001,
        'memory_size': 50000,
        'batch_size': 128,
        'eps_start': 1.0, 
        'eps_end': 0.01, 
        'eps_decay': 1000,
        'num_episodes': 140,
        'target_update_freq': 100,
        'seed': 42
    },
    'Acrobot-v1': {
        'gamma': 0.999, 
        'learning_rate': 2e-4, 
        'memory_size': 50000, 
        'batch_size': 128, 
        'eps_start': 1.0, 
        'eps_end': 0.01, 
        'eps_decay': 1000, 
        'num_episodes': 700,     
        'target_update_freq': 500, 
        'seed': 42 
    },
    'MountainCar-v0': {
        'gamma': 0.9999, 
        'learning_rate': 1e-3, 
        'memory_size': 50000, 
        'batch_size': 128, 
        'eps_start': 1.0, 
        'eps_end': 0.05, 
        'eps_decay': 20000, 
        'num_episodes': 1500,
        'target_update_freq': 500, 
        'seed': 100 
    },
    'Pendulum-v1': {
        'gamma': 0.99,
        'learning_rate': 1e-4,
        'memory_size': 500000,
        'batch_size': 128,           
        'eps_start': 1.0,
        'eps_end': 0.01,
        'eps_decay': 0.001,
        'num_episodes': 2000,
        'target_update_freq': 500,
        'seed': 100,
    },
}

# SWEEP CONFIGURATIONS 
SWEEP_VARIATIONS = {
    'DQN_BASE': {'is_ddqn': False},
    'DDQN_BASE': {'is_ddqn': True},

    # # GAMMA SWEEP (Test effect of Discount Factor)
    # 'DDQN_GAMMA_0.9': {'is_ddqn': True, 'gamma': 0.9},
    # 'DDQN_GAMMA_0.9999': {'is_ddqn': True, 'gamma': 0.9999},

    # # LEARNING RATE SWEEP (Test effect of NN Learning Rate)
    # 'DDQN_LR_1e-5': {'is_ddqn': True, 'learning_rate': 1e-5}, 
    # 'DDQN_LR_1e-3': {'is_ddqn': True, 'learning_rate': 1e-3},

    # # EPSILON DECAY SWEEP (Test effect of Exploration Speed)
    # 'DDQN_EPS_FAST': {'is_ddqn': True, 'eps_decay': 500}, # Fast decay (less exploration)
    # 'DDQN_EPS_SLOW': {'is_ddqn': True, 'eps_decay': 50000}, # Slow decay (more exploration)
}


def run_experiment(env_name, config):
    """Sets up W&B, trains, tests, and logs final results."""
    
    # W&B Setup
    run_name = f"{env_name}_{config['name']}"
    wandb.init(project="CMPS458_DQN_DDQN_Assignment", config=config, name=run_name, reinit=True)
    
    # Environment Setup
    temp_env = gym.make(env_name)
    state_dim = temp_env.observation_space.shape[0]
    
    if 'Pendulum' in env_name:
        action_dim = 5 
    elif isinstance(temp_env.action_space, gym.spaces.Discrete):
        action_dim = temp_env.action_space.n
    else:
        raise ValueError(f"Action space of {env_name} is not discrete. DQN requires discretization.")
        
    temp_env.close()

    # Initialize Agent
    agent = DQNAgent(state_dim, action_dim, DEVICE, config['is_ddqn'], config)

    # 1. Training Phase
    print(f"\n--- Starting Training: {run_name} ---")
    trained_agent = train_agent(env_name, agent, config, config['num_episodes'], log_wandb=True)

    # 2. Testing Phase (100 tests for stability)
    print(f"\n--- Starting Testing: {run_name} (100 trials) ---")
    avg_duration, std_duration, test_durations, avg_reward, std_reward = test_agent(env_name, trained_agent, num_tests=100, record_video=True)
    
    duration_data = [[duration] for duration in test_durations]
    duration_table = wandb.Table(data=duration_data, columns=["Test Duration (Steps)"])
    
    # Log final results to W&B
    wandb.log({
        "test/avg_episode_duration": avg_duration,
        "test/std_episode_duration": std_duration,
        "test/avg_reward": avg_reward,
        "test/std_reward": std_reward,
        "test/durations_table": duration_table,  
        "Final_Status": "Completed",
        "Algorithm": "DDQN" if config['is_ddqn'] else "DQN"
    })
    
    # Print results to console
    print(f"\nRESULTS: {run_name}")
    print(f"  Average Test Duration (100 trials): {avg_duration:.2f}")
    print(f"  Stability on Duration (Std Dev): {std_duration:.2f}")
    print(f"  Average Test Reward (100 trials): {avg_reward:.2f}")
    print(f"  Stability on Reward (Std Dev): {std_reward:.2f}")

    wandb.finish()


if __name__ == "__main__":
    
    # Iterate through all four environments
    for env_name in ENVIRONMENTS:
        base_config = ENV_BASE_HYPERPARAMS[env_name]
        
        # Iterate through all sweep variations 
        for sweep_name, sweep_params in SWEEP_VARIATIONS.items():
            
            final_config = base_config.copy()
            final_config.update(sweep_params)
            final_config['name'] = sweep_name 
            
            # Run the experiment
            run_experiment(env_name, final_config)