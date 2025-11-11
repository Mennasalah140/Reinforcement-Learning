import os
import torch
import wandb
import gymnasium as gym # Required for checking action space type
from train_utils import train_agent, test_agent, DEVICE
from dqn_agent import DQNAgent

# --- Define Experiments ---
ENVIRONMENTS = [
    'CartPole-v1', 
    'Acrobot-v1', 
    'MountainCar-v0', 
    'Pendulum-v1' 
]

# Base Hyperparameters (Increased episodes for harder environments)
BASE_HYPERPARAMS = {
    'gamma': 0.99,
    'learning_rate': 1e-4,
    'memory_size': 10000,
    'batch_size': 64,
    'eps_start': 0.9,
    'eps_end': 0.05,
    'eps_decay': 10000, # Increased decay steps for more exploration
    'num_episodes': 100 # Increased episodes for harder environments
}

# --- Define Hyperparameter Sweep Configurations ---
SWEEP_CONFIGS = [
    # BASELINES
    {'name': 'DQN_BASE', 'is_ddqn': False, **BASE_HYPERPARAMS},
    {'name': 'DDQN_BASE', 'is_ddqn': True, **BASE_HYPERPARAMS},
    
    # # TUNING FOR Q3: Example: Test effect of Gamma (Discount Factor)
    # {'name': 'DDQN_GAMMA_0.8', 'is_ddqn': True, **{k:v for k,v in BASE_HYPERPARAMS.items() if k!='gamma'}, 'gamma': 0.8},
    # {'name': 'DDQN_GAMMA_0.999', 'is_ddqn': True, **{k:v for k,v in BASE_HYPERPARAMS.items() if k!='gamma'}, 'gamma': 0.999},

    # # TUNING FOR Q3: Example: Test effect of Learning Rate
    # {'name': 'DDQN_LR_1e-3', 'is_ddqn': True, **{k:v for k,v in BASE_HYPERPARAMS.items() if k!='learning_rate'}, 'learning_rate': 1e-3},
]


def run_experiment(env_name, config):
    """Sets up W&B, trains, tests, and logs final results."""
    
    # W&B Setup
    run_name = f"{env_name}_{config['name']}"
    wandb.init(project="CMPS458_DQN_DDQN_Assignment", config=config, name=run_name, reinit=True)
    
    # Environment Setup
    temp_env = gym.make(env_name)
    state_dim = temp_env.observation_space.shape[0]
    
    # Determine Action Dimension (5 for Pendulum, otherwise use env.action_space.n)
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

    # 2. Testing Phase (100 tests for stability, Q2)
    print(f"\n--- Starting Testing: {run_name} (100 trials) ---")
    avg_duration, std_duration = test_agent(env_name, trained_agent, num_tests=100, record_video=True)
    
    # Log final results to W&B
    wandb.log({
        "test/avg_episode_duration": avg_duration,
        "test/std_episode_duration": std_duration,
        "Final_Status": "Completed",
        "Algorithm": "DDQN" if config['is_ddqn'] else "DQN"
    })
    
    # Print results to console (for Q2)
    print(f"\nRESULTS: {run_name}")
    print(f"  Average Test Duration (100 trials): {avg_duration:.2f}")
    print(f"  Stability (Std Dev): {std_duration:.2f}")
    
    wandb.finish()


if __name__ == "__main__":
    # Note: Ensure you have installed all required libraries: gymnasium, pytorch, wandb, numpy
    
    # You must run all sweeps to answer Q3!
    for env_name in ENVIRONMENTS:
        for config in SWEEP_CONFIGS:
            run_experiment(env_name, config)