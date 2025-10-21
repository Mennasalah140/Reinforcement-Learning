import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import time
import os

from grid_maze_env import GridMazeEnv
from policy_iteration import (
    policy_iteration_solver, state_to_coords, coords_to_state
)

# --- Define a Fixed Maze Configuration (Needed for PI training) ---
# NOTE: To answer Q3, you should run PI on this configuration.
START_POS = (0, 0) # Fixed starting point for applying the learned policy
GOAL_POS = (4, 4)
BAD_CELL_1 = (1, 1)
BAD_CELL_2 = (3, 3)
FIXED_CONFIG_COORDS = (START_POS, GOAL_POS, BAD_CELL_1, BAD_CELL_2)

def train_and_run_policy():
    print("--- 1. Running Policy Iteration (PI) ---")
    
    # Policy Iteration solves the MDP based on the fixed Goal and Bad cells
    V_final, policy_final, iterations = policy_iteration_solver(
        goal_coords=GOAL_POS,
        bad_coords=[BAD_CELL_1, BAD_CELL_2],
        gamma=0.99,
        theta=1e-6
    )

    print(f"PI Converged in {iterations} iterations.")
    
    # Optional: Display the learned policy (0:R, 1:U, 2:L, 3:D)
    policy_map = policy_final.reshape(5, 5)
    print("\nLearned Policy Map (0:R, 1:U, 2:L, 3:D):\n", policy_map)

    print("\n--- 2. Applying Learned Policy to Gym Environment and Recording ---")
    
    # 2. Setup Gym Environment and Video Recording
    video_dir = "./videos"
    os.makedirs(video_dir, exist_ok=True)
    
    # Initialize the custom environment, must use 'rgb_array' for video recording
    env = GridMazeEnv(render_mode="rgb_array")
    
    # Apply the RecordVideo Wrapper (Task 6)
    env = RecordVideo(
        env, 
        video_folder=video_dir, 
        episode_trigger=lambda x: x == 0, # Record the first episode
        name_prefix="PolicyIteration_Agent"
    )

    # 3. Run the Agent using the Learned Policy
    
    # Reset the environment, forcing it to use the fixed configuration used for PI
    obs, info = env.reset(seed=42, options={'fixed_config': FIXED_CONFIG_COORDS})
    
    done = False
    total_reward = 0
    step = 0
    
    while not done:
        # Get agent's current coordinates from the observation
        agent_x, agent_y = obs[0], obs[1]
        
        # Convert coordinates to the 25-state index
        state = coords_to_state(agent_x, agent_y)
        
        # Use the learned policy to select the action
        action = policy_final[state] 
        
        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated
        total_reward += reward
        step += 1
        
        # Safety break for long episodes (e.g., if agent gets stuck)
        if step > 200:
            print("Episode truncated after 200 steps.")
            break

    print(f"\nEpisode finished.")
    print(f"Total Steps: {step}")
    print(f"Final Reward: {total_reward}")
    print(f"Video recorded in: {env.video_folder}")

    env.close()

if __name__ == "__main__":
    train_and_run_policy()