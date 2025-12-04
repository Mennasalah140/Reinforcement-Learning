import os
import torch
import wandb
import gymnasium as gym
from train_utils_pg import train_agent_pg, test_agent, DEVICE
from a2c_agent import A2CAgent 
from ppo_agent import PPOAgent
from sac_agent import SACAgent 

ENVIRONMENTS = {
    # 'CartPole-v1': {'discrete': True, 'action_dim': 2, 'torque': 1.0},
    # 'Acrobot-v1': {'discrete': True, 'action_dim': 3, 'torque': 1.0},
    'MountainCar-v0': {'discrete': True, 'action_dim': 3, 'torque': 1.0},
    # 'Pendulum-v1': {'discrete': False, 'action_dim': 1, 'torque': 2.0} 
}

# HYPERPARAMETERS
ALGO_HYPERPARAMS = {
    # No epsilon decay, as they use policy gradient not value based so it's stochastic which does exploration by default
    'CartPole-v1': {
        'A2C': { 
            # No memory size, as A2C uses on-policy updates, so data relies on current policy
            # No batch size, as A2C updates after each trajectory, doesn't use mini-batches
            'gamma': 0.999,
            'learning_rate': 5e-3,
            'trajectory_size': 4096,
            'num_episodes': 500,
        },

        'PPO': {
            # Smaller batch size for faster updates, to maximize use of trajectory data
            'gamma': 0.995,
            'learning_rate': 3e-4,
            'trajectory_size': 1024,
            'num_episodes': 500,
            'clip_epsilon': 0.2, 
            'ppo_epochs': 15,    
            'minibatch_size': 64,
            'entropy_coeff': 0.005,
        },

        'SAC': {
            'gamma': 0.995,
            'learning_rate': 3e-4,
            'num_episodes': 300,
            'memory_size': 25000, 
            'batch_size': 64,
            'tau': 0.005,         
            'alpha_start': 0.05,  
        }
    },
    'Acrobot-v1': {
        'A2C': {
            'gamma': 0.995,           
            'learning_rate': 1e-4,    
            'trajectory_size': 8192,  
            'num_episodes': 1500,     
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
            'num_episodes': 1000,
            'memory_size': 50000,
            'batch_size': 128,
            'tau': 0.005,
            'alpha_start': 0.2
        }
    },
    'MountainCar-v0': {
        'A2C': {
            'gamma': 0.999,              
            'learning_rate': 3e-4,        
            'trajectory_size': 8192,     
            'num_episodes': 1500,         
        },
        'PPO': {
            'gamma': 0.999,
            'learning_rate': 2e-4,
            'trajectory_size': 4096,
            'num_episodes': 1500,
            'clip_epsilon': 0.2,
            'ppo_epochs': 10,
            'minibatch_size': 64,
            'entropy_coeff': 0.01         
        },
        'SAC': {
            'gamma': 0.999,
            'learning_rate': 3e-4,        
            'num_episodes': 1500,
            'memory_size': 50000,
            'batch_size': 256,            
            'tau': 0.005,
            'alpha_start': 0.2
        }
    },

    'Pendulum-v1': {
        'A2C': {
            'gamma': 0.995,
            'learning_rate': 3e-4,
            'trajectory_size': 2048,
            'num_episodes': 1000,
        },
        'PPO': {
            'gamma': 0.995,
            'learning_rate': 3e-4,
            'trajectory_size': 2048,
            'num_episodes': 1000,
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


SWEEP_VARIATIONS = {
    # Compare A2C, PPO, SAC
    # 'A2C_BASE': {'algorithm': 'A2C'},
    # 'PPO_BASE': {'algorithm': 'PPO'}, 
    'SAC_BASE': {'algorithm': 'SAC'}, 
}


def create_agent(algorithm_name, env_details, config):
    """Function to create the correct agent instance."""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env_details['action_dim']
    is_discrete = env_details['discrete']
    torque = env_details['torque']
    env.close()

    if algorithm_name == 'A2C':
        agent = A2CAgent(state_dim, action_dim, DEVICE, config, is_discrete)
    elif algorithm_name == 'PPO':
        agent = PPOAgent(state_dim, action_dim, DEVICE, config, is_discrete) 
    elif algorithm_name == 'SAC':
        agent = SACAgent(state_dim, action_dim, DEVICE, config, is_discrete)
    else:
        return None
        
    if not is_discrete:
        agent.action_scale = torque
        
    return agent

def run_experiment(env_name, config):    
    # W&B Setup
    run_name = f"{env_name}_{config['name']}"
    wandb.init(project="CMPS458_PolicyGradient_Assignment", config=config, name=run_name, reinit=True)
    
    # Environment Details
    env_details = ENVIRONMENTS[env_name]

    # Initialize Agent
    agent = create_agent(config['algorithm'], env_details, config)

    # Training Phase
    print(f"\n--- Starting Training: {run_name} ---")
    trained_agent = train_agent_pg(env_name, agent, config, config['num_episodes'], log_wandb=True)

    # Testing Phase (100 tests) 
    print(f"\n--- Starting Testing: {run_name} (100 trials) ---")
    avg_duration, std_duration, test_durations, avg_reward, std_reward, test_rewards = test_agent(env_name, trained_agent, num_tests=100, record_video=True)
    
    duration_data = [[duration] for duration in test_rewards]
    duration_table = wandb.Table(data=duration_data, columns=["Test Reward"])
    
    # Log final results to W&B
    wandb.log({
        "test/avg_episode_duration": avg_duration,
        "test/std_episode_duration": std_duration,
        "test/durations_table": duration_table,
        "test/avg_episode_reward": avg_reward,
        "test/std_episode_reward": std_reward,
        "Final_Status": "Completed",
        "Algorithm": config['algorithm']
    })
    
    print(f"\nRESULTS: {run_name}")
    print(f"  Average Test Duration (100 trials): {avg_duration:.2f}")
    print(f"  Stability (Std Dev): {std_duration:.2f}")
    
    wandb.finish()


if __name__ == "__main__":
    
    for env_name in ENVIRONMENTS:
        
        for sweep_name, sweep_params in SWEEP_VARIATIONS.items():
            
            algo = sweep_params['algorithm']
            base_config = ALGO_HYPERPARAMS[env_name][algo].copy() 
            
            final_config = base_config.copy()
            final_config.update(sweep_params)
            final_config['name'] = sweep_name 
            final_config['env_name'] = env_name
            final_config['seed'] = 42 
            
            run_experiment(env_name, final_config)
            pass