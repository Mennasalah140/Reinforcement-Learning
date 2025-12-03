import os
import torch
import wandb
import gymnasium as gym
from train_utils_pg import train_agent_pg, test_agent, DEVICE
from a2c_agent import A2CAgent 
from ppo_agent import PPOAgent
from sac_agent import SACAgent 

# --- Define Experiments ---
ENVIRONMENTS = {
    # 'CartPole-v1': {'discrete': True, 'action_dim': 2},
    # 'Acrobot-v1': {'discrete': True, 'action_dim': 3},
    'MountainCar-v0': {'discrete': True, 'action_dim': 3},
    # Pendulum is Continuous but we treat it as discrete for A2C/PPO. SAC runs continuous (sac_discrete=False).
    'Pendulum-v1': {'discrete': True, 'action_dim': 5, 'sac_discrete': False} 
}

# --- ALGORITHM-SPECIFIC HYPERPARAMETERS (Optimized for each environment) ---
ALGO_HYPERPARAMS = {
    'CartPole-v1': {
        'A2C': { #random keeps oscilating each time
            'gamma': 0.999,
            'learning_rate': 5e-3,
            'trajectory_size': 4096,
            'num_episodes': 500,
        },

        'PPO': {
            'gamma': 0.99,
            'learning_rate': 3e-4,
            'trajectory_size': 4096,
            'num_episodes': 600,
            'clip_epsilon': 0.2,
            'ppo_epochs': 10,
            'minibatch_size': 64,
            'entropy_coeff': 0.01
        },

        'SAC': {
            'gamma': 0.99,
            'learning_rate': 3e-4,
            'num_episodes': 500,
            'memory_size': 100000,
            'batch_size': 128,
            'tau': 0.005,
            'alpha_start': 0.2,
        }
    },
    'Acrobot-v1': {
        'A2C': {
            'gamma': 0.95,           # higher gamma to handle delayed reward
            'learning_rate': 7e-4,    # lower LR for stability
            'trajectory_size': 8192,  # longer trajectories for better gradient estimates
            'num_episodes': 1500,     # more episodes to allow exploration
        },
        'PPO': {
            'gamma': 0.995,
            'learning_rate': 3e-4,
            'trajectory_size': 8192,
            'num_episodes': 1000,
            'clip_epsilon': 0.2,
            'ppo_epochs': 10,
            'minibatch_size': 128,
            'entropy_coeff': 0.01
        },
        'SAC': {
            'gamma': 0.995,
            'learning_rate': 5e-4,
            'num_episodes': 500,
            'memory_size': 50000,
            'batch_size': 128,
            'tau': 0.005,
            'alpha_start': 0.2
        }
    },
  'MountainCar-v0': {
    'A2C': {
        'gamma': 0.999,               # long-term reward propagation
        'learning_rate': 3e-4,        # slightly higher for faster learning
        'trajectory_size': 8192,      # longer trajectories for sparse rewards
        'num_episodes': 1200,         # more episodes to converge
    },
    'PPO': {
        'gamma': 0.999,
        'learning_rate': 2e-4,
        'trajectory_size': 4096,
        'num_episodes': 600,
        'clip_epsilon': 0.2,
        'ppo_epochs': 10,
        'minibatch_size': 64,
        'entropy_coeff': 0.01         # helps exploration in sparse rewards
    },
    'SAC': {
        'gamma': 0.999,
        'learning_rate': 3e-4,        # slightly higher for faster learning
        'num_episodes': 600,
        'memory_size': 50000,
        'batch_size': 256,            # larger batch for more stable updates
        'tau': 0.005,
        'alpha_start': 0.2
    }
    },

    'Pendulum-v1': {
        'A2C': {
            'gamma': 0.995,
            'learning_rate': 3e-4,
            'trajectory_size': 2048,
            'num_episodes': 700,
        },
        'PPO': {
            'gamma': 0.995,
            'learning_rate': 3e-4,
            'trajectory_size': 2048,
            'num_episodes': 700,
            'clip_epsilon': 0.2,
            'ppo_epochs': 10,
            'minibatch_size': 64
        },
        'SAC': {
            'gamma': 0.995,
            'learning_rate': 3e-4,
            'num_episodes': 800,
            'memory_size': 100000,
            'batch_size': 128,
            'tau': 0.005,
            'alpha_start': 0.2
        }
    }
}


# --- SWEEP CONFIGURATIONS (Assignment Q3) ---
SWEEP_VARIATIONS = {
    # 1. BASELINES (Compare A2C, PPO, SAC)
    # 'A2C_BASE': {'algorithm': 'A2C'},
    'PPO_BASE': {'algorithm': 'PPO'}, 
    # 'SAC_BASE': {'algorithm': 'SAC'}, 
}


def create_agent(algorithm_name, env_details, config):
    """Factory function to create the correct agent instance."""
    env_name = config['env_name'] # Retrieve env_name from config for agent instantiation
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env_details['action_dim']
    is_discrete = env_details['discrete']
    env.close()

    if algorithm_name == 'A2C':
        return A2CAgent(state_dim, action_dim, DEVICE, config, is_discrete)
    elif algorithm_name == 'PPO':
        return PPOAgent(state_dim, action_dim, DEVICE, config, is_discrete) 
    elif algorithm_name == 'SAC':
        # For SAC, action_dim is 1 because the policy network outputs 1 continuous torque value.
        sac_action_dim = 1
        is_discrete = env_details.get('sac_discrete', is_discrete)
        return SACAgent(state_dim, sac_action_dim, DEVICE, config, is_discrete)
    return None

def run_experiment(env_name, config):
    """Sets up W&B, trains, tests, and logs final results."""
    
    # W&B Setup
    run_name = f"{env_name}_{config['name']}"
    wandb.init(project="CMPS458_PolicyGradient_Assignment", config=config, name=run_name, reinit=True)
    
    # Environment Details
    env_details = ENVIRONMENTS[env_name]

    # Initialize Agent
    agent = create_agent(config['algorithm'], env_details, config)

    # 1. Training Phase
    print(f"\n--- Starting Training: {run_name} ---")
    trained_agent = train_agent_pg(env_name, agent, config, config['num_episodes'], log_wandb=True)

    # 2. Testing Phase (100 tests for stability, Q2) [cite: 17]
    print(f"\n--- Starting Testing: {run_name} (100 trials) ---")
    avg_duration, std_duration, test_durations = test_agent(env_name, trained_agent, num_tests=100, record_video=True)
    
    # --- Logging Fix (Assignment Q2 Figure) ---
    duration_data = [[duration] for duration in test_durations]
    duration_table = wandb.Table(data=duration_data, columns=["Test Duration (Steps)"])
    
    # Log final results to W&B
    wandb.log({
        "test/avg_episode_duration": avg_duration,
        "test/std_episode_duration": std_duration,
        "test/durations_table": duration_table,
        "Final_Status": "Completed",
        "Algorithm": config['algorithm']
    })
    
    # Print results to console
    print(f"\nRESULTS: {run_name}")
    print(f"  Average Test Duration (100 trials): {avg_duration:.2f}")
    print(f"  Stability (Std Dev): {std_duration:.2f}")
    
    wandb.finish()


if __name__ == "__main__":
    
    for env_name in ENVIRONMENTS:
        
        for sweep_name, sweep_params in SWEEP_VARIATIONS.items():
            
            algo = sweep_params['algorithm']
            base_config = ALGO_HYPERPARAMS[env_name][algo].copy() # Get algorithm-specific config
            
            final_config = base_config.copy()
            final_config.update(sweep_params)
            final_config['name'] = sweep_name 
            final_config['env_name'] = env_name # Add environment name for retrieval in create_agent
            final_config['seed'] = 42 # Use a fixed seed for reproducible baselines
            
            run_experiment(env_name, final_config)
            pass