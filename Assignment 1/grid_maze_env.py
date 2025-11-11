import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

# --- Environment Constants ---
GRID_SIZE = 5
CELL_COUNT = GRID_SIZE * GRID_SIZE
WINDOW_SIZE = 500
CELL_PIXEL_SIZE = WINDOW_SIZE // GRID_SIZE

# --- Reward Function Design (Your choice for the assignment) ---
# Reason: Small negative reward encourages shortest path; large terminal rewards define goal/danger.
DEFAULT_REWARD = -1       # Cost per step 
GOAL_REWARD = 100         # High positive reward for reaching the goal (G)
BAD_CELL_REWARD = -100    # High negative punishment for hitting a bad cell (X)

class GridMazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # Action Space: {0: right, 1: up, 2: left, 3: down}
        self.action_space = spaces.Discrete(4)
        
        # Observation Space: (S_x, S_y, G_x, G_y, X1_x, X1_y, X2_x, X2_y)
        low = np.array([0] * 8, dtype=np.int32)
        high = np.array([GRID_SIZE - 1] * 8, dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        self.agent_pos = None
        self.goal_pos = None
        self.bad_cells = []
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        """Returns the current observation array (8 coordinates)"""
        return np.array([
            *self.agent_pos,
            *self.goal_pos,
            *self.bad_cells[0],
            *self.bad_cells[1]
        ], dtype=np.int32)

    def _get_info(self):
        """Returns auxiliary information"""
        return {"distance_to_goal": np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal_pos))}

    def reset(self, seed=None, options=None):
        """Resets the environment to a new random configuration"""
        super().reset(seed=seed)
        
        # 1. Generate 4 distinct random coordinates for S, G, X1, X2
        all_cells = np.arange(CELL_COUNT)
        self.np_random.shuffle(all_cells)
        
        def to_coords(flat_index):
            x = flat_index % GRID_SIZE
            y = flat_index // GRID_SIZE
            return (x, y)

        self.agent_pos = to_coords(all_cells[0])
        self.goal_pos = to_coords(all_cells[1])
        self.bad_cells = [to_coords(all_cells[2]), to_coords(all_cells[3])]

        # If a fixed maze configuration is provided via options, override random generation (used for PI application)
        if options and 'fixed_config' in options:
            s, g, x1, x2 = options['fixed_config']
            self.agent_pos = s
            self.goal_pos = g
            self.bad_cells = [x1, x2]

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _get_stochastic_next_pos(self, action):
        """
        Calculates the next position based on the 70%/15%/15% stochastic movement rule.
        """
        # (dx, dy) for the intended action (70%), and the two perpendicular actions (15% each)
        movements = {
            0: [(1, 0), (0, 1), (0, -1)],  # Right
            1: [(0, 1), (1, 0), (-1, 0)],  # Up
            2: [(-1, 0), (0, 1), (0, -1)], # Left
            3: [(0, -1), (1, 0), (-1, 0)]  # Down
        }
        
        probabilities = [0.70, 0.15, 0.15]

        # Sample the actual move based on probabilities
        move_idx = self.np_random.choice([0, 1, 2], p=probabilities)
        dx, dy = movements[action][move_idx]

        # Calculate new coordinates and clamp them within [0, GRID_SIZE-1]
        new_x = np.clip(self.agent_pos[0] + dx, 0, GRID_SIZE - 1)
        new_y = np.clip(self.agent_pos[1] + dy, 0, GRID_SIZE - 1)
        
        return (new_x, new_y)

    def _calculate_outcome(self):
        """Determines reward and termination status"""
        terminated = False
        reward = DEFAULT_REWARD 
        
        if self.agent_pos == self.goal_pos:
            reward = GOAL_REWARD
            terminated = True
        elif self.agent_pos in self.bad_cells:
            reward = BAD_CELL_REWARD
            terminated = True
            
        return reward, terminated

    def step(self, action):
        """Executes one step in the environment"""
        next_pos = self._get_stochastic_next_pos(action)
        self.agent_pos = next_pos
        
        reward, terminated = self._calculate_outcome()
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info # last False is for truncation

    # --- PyGame Rendering Methods ---
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            self.font = pygame.font.SysFont(None, 24)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        canvas.fill((255, 255, 255)) 

        COLOR_AGENT = (0, 0, 255)
        COLOR_GOAL = (0, 255, 0)
        COLOR_BAD = (255, 0, 0)
        COLOR_GRID = (200, 200, 200)

        # Draw Grid Lines
        for x in range(GRID_SIZE + 1):
            pygame.draw.line(canvas, COLOR_GRID, (x * CELL_PIXEL_SIZE, 0), (x * CELL_PIXEL_SIZE, WINDOW_SIZE), 2)
            pygame.draw.line(canvas, COLOR_GRID, (0, x * CELL_PIXEL_SIZE), (WINDOW_SIZE, x * CELL_PIXEL_SIZE), 2)

        def get_rect(coords):
            x, y = coords
            # Map (x, y) coordinates to screen position. Assuming (0,0) is top-left.
            return pygame.Rect(x * CELL_PIXEL_SIZE, y * CELL_PIXEL_SIZE, CELL_PIXEL_SIZE, CELL_PIXEL_SIZE)

        # Draw Goal (G)
        pygame.draw.rect(canvas, COLOR_GOAL, get_rect(self.goal_pos))
        text_g = self.font.render('G', True, (0, 0, 0))
        canvas.blit(text_g, get_rect(self.goal_pos).move(CELL_PIXEL_SIZE // 4, CELL_PIXEL_SIZE // 4))
        
        # Draw Bad Cells (X)
        for bad_pos in self.bad_cells:
            pygame.draw.rect(canvas, COLOR_BAD, get_rect(bad_pos))
            text_x = self.font.render('X', True, (0, 0, 0))
            canvas.blit(text_x, get_rect(bad_pos).move(CELL_PIXEL_SIZE // 4, CELL_PIXEL_SIZE // 4))
            
        # Draw Agent (S)
        center_x = self.agent_pos[0] * CELL_PIXEL_SIZE + CELL_PIXEL_SIZE // 2
        center_y = self.agent_pos[1] * CELL_PIXEL_SIZE + CELL_PIXEL_SIZE // 2
        pygame.draw.circle(canvas, COLOR_AGENT, (center_x, center_y), CELL_PIXEL_SIZE // 3)
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            
        return np.transpose(np.array(pygame.surfarray.array3d(canvas)), (1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()