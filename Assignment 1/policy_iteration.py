import numpy as np
from grid_maze_env import (
    GRID_SIZE, CELL_COUNT, DEFAULT_REWARD, GOAL_REWARD, BAD_CELL_REWARD
)

NUM_ACTIONS = 4 # {0: right, 1: up, 2: left, 3: down}

# Pre-calculate the deterministic change vector for each action
ACTION_DELTAS = {
    0: (1, 0),  # Right
    1: (0, 1),  # Up
    2: (-1, 0), # Left
    3: (0, -1)  # Down
}

# --- Helper Functions for 25-State MDP ---

def coords_to_state(x, y):
    """Map (x, y) coordinates to a single state index (0 to 24)"""
    return y * GRID_SIZE + x

def state_to_coords(s):
    """Map state index (0 to 24) to (x, y) coordinates"""
    return (s % GRID_SIZE, s // GRID_SIZE)

def get_possible_transitions(s, action):
    """
    Calculates the transition probabilities and next states for a fixed MDP.
    Returns: list of (probability, next_state_index, reward, is_terminal).
    """
    x, y = state_to_coords(s)
    
    # 70% intended, 15% perpendicular 1, 15% perpendicular 2
    intended_delta = ACTION_DELTAS[action]
    
    if intended_delta == (1, 0) or intended_delta == (-1, 0): # Horizontal move
        perp_deltas = [(0, 1), (0, -1)]
    else: # Vertical move
        perp_deltas = [(1, 0), (-1, 0)]
        
    moves = [intended_delta, *perp_deltas]
    probabilities = [0.70, 0.15, 0.15]
    
    transitions = []
    
    for (dx, dy), prob in zip(moves, probabilities):
        # Calculate new coordinates, clamping within [0, 4]
        next_x = np.clip(x + dx, 0, GRID_SIZE - 1)
        next_y = np.clip(y + dy, 0, GRID_SIZE - 1)
        next_s = coords_to_state(next_x, next_y)
        
        # NOTE: Reward calculation is NOT done here for true MDP P-matrix, 
        # but is done in PI function for state-action-state' triple.
        transitions.append((prob, next_s))
        
    return transitions

def calculate_reward(s_prime, goal_state, bad_states):
    """Calculates the reward for reaching state s_prime"""
    if s_prime == goal_state:
        return GOAL_REWARD
    elif s_prime in bad_states:
        return BAD_CELL_REWARD
    else:
        return DEFAULT_REWARD

def is_terminal(s_prime, goal_state, bad_states):
    """Checks if the next state is a terminal state"""
    return s_prime == goal_state or s_prime in bad_states

# --- Policy Iteration Algorithm ---

def policy_evaluation(V, policy, goal_state, bad_states, gamma, theta):
    """Performs Policy Evaluation (Iterative V calculation)"""
    
    while True:
        delta = 0
        for s in range(CELL_COUNT):
            # Terminal states have V(s) = 0
            if is_terminal(s, goal_state, bad_states):
                continue
                
            v_old = V[s]
            a = policy[s] # action from current policy
            
            v_new = 0
            
            # V(s) = Sum_{s'} P(s'|s, a) * [R(s') + gamma * V(s')]
            transitions = get_possible_transitions(s, a)
            
            for prob, s_prime in transitions:
                reward = calculate_reward(s_prime, goal_state, bad_states)
                
                # If s' is terminal, V(s') is effectively 0 for the Bellman update
                v_prime = 0 if is_terminal(s_prime, goal_state, bad_states) else V[s_prime]
                
                v_new += prob * (reward + gamma * v_prime)
            
            V[s] = v_new
            delta = max(delta, np.abs(v_old - V[s]))
        
        if delta < theta:
            break
            
    return V

def policy_improvement(V, policy, goal_state, bad_states, gamma):
    """Performs Policy Improvement (Greedy policy update)"""
    policy_stable = True
    
    for s in range(CELL_COUNT):
        # Skip improvement for terminal states
        if is_terminal(s, goal_state, bad_states):
            continue

        old_action = policy[s]
        
        # Find the best action a* = argmax_a Q(s, a)
        q_values = np.zeros(NUM_ACTIONS)
        for a in range(NUM_ACTIONS):
            # Q(s, a) = Sum_{s'} P(s'|s, a) * [R(s') + gamma * V(s')]
            transitions = get_possible_transitions(s, a)

            for prob, s_prime in transitions:
                reward = calculate_reward(s_prime, goal_state, bad_states)
                v_prime = 0 if is_terminal(s_prime, goal_state, bad_states) else V[s_prime]
                q_values[a] += prob * (reward + gamma * v_prime)

        new_action = np.argmax(q_values)
        policy[s] = new_action
        
        if new_action != old_action:
            policy_stable = False
            
    return policy, policy_stable

def policy_iteration_solver(goal_coords, bad_coords, gamma=0.99, theta=1e-6):
    """
    Policy Iteration main loop. Returns optimal V, optimal policy, and iteration count.
    """
    # Map fixed (x, y) coordinates to state indices
    goal_state = coords_to_state(*goal_coords)
    bad_states = [coords_to_state(*bc) for bc in bad_coords]
    
    V = np.zeros(CELL_COUNT)            # Initialize V(s)
    policy = np.zeros(CELL_COUNT, dtype=int) # Initialize policy (all 'right'=0)
    
    iteration_count = 0
    while True:
        # 1. Policy Evaluation
        V = policy_evaluation(V, policy, goal_state, bad_states, gamma, theta)
        
        # 2. Policy Improvement
        policy, policy_stable = policy_improvement(V, policy, goal_state, bad_states, gamma)
        
        iteration_count += 1
        
        if policy_stable:
            return V, policy, iteration_count