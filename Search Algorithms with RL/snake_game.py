import pygame
import numpy as np
import random
from enum import Enum
from collections import deque


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class SnakeGame:
    def __init__(self, width=20, height=15, cell_size=20):
        self.width = width
        self.height = height
        self.cell_size = cell_size

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.DARK_GREEN = (0, 128, 0)
        self.GREY = (100, 100, 100)

        # Pygame setup (lazy init)
        self.screen = None
        self.clock = None
        self.font = None

        # Game state
        self.snake = []
        self.direction = Direction.RIGHT
        self.food = None
        self.score = 0
        self.game_over = False

        # Obstacles: list of (x, y)
        self.obstacles = []

        # Step count & recent positions (for loop detection)
        self.steps = 0
        self.last_positions = deque(maxlen=80)  # recent head positions

        self.reset()

    def reset(self):
        """Reset the game to initial state"""
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = Direction.RIGHT
        self.score = 0
        self.game_over = False
        self.steps = 0
        self.last_positions.clear()
        self.last_positions.append(self.snake[0])

        # New random obstacles every game
        self._create_obstacles()

        # Food not on snake or obstacles
        self.food = self._generate_food()

        return self._get_observation()

    def _create_obstacles(self):
        """Create random scattered obstacles for this game."""
        self.obstacles = []
        max_cells = self.width * self.height
        num_obstacles = max(50, max_cells // 150)  # tune if you like

        attempts = 0
        while len(self.obstacles) < num_obstacles and attempts < num_obstacles * 20:
            attempts += 1
            ox = random.randint(0, self.width - 1)
            oy = random.randint(0, self.height - 1)
            pos = (ox, oy)

            # Avoid starting cell and duplicates
            if pos == self.snake[0]:
                continue
            if pos in self.obstacles:
                continue

            self.obstacles.append(pos)

    def _generate_food(self):
        """Generate food at random position not occupied by snake or obstacles."""
        while True:
            fx = random.randint(0, self.width - 1)
            fy = random.randint(0, self.height - 1)
            food = (fx, fy)
            if food not in self.snake and food not in self.obstacles:
                return food

    def _get_observation(self):
        """
        Improved Observation:
        - 11 boolean features (danger, direction, food location)
        - 4 coordinate features (head x/y, tail x/y) normalized
        Total: 15 features
        """
        if not self.snake:
            return np.zeros(15, dtype=np.float32)

        head = self.snake[0]
        tail = self.snake[-1]

        directions = [Direction.UP, Direction.RIGHT,
                      Direction.DOWN, Direction.LEFT]
        current_idx = directions.index(self.direction)

        danger_straight = self._is_collision(head, self.direction)
        danger_right = self._is_collision(
            head, directions[(current_idx + 1) % 4])
        danger_left = self._is_collision(
            head, directions[(current_idx - 1) % 4])

        dir_up = self.direction == Direction.UP
        dir_right = self.direction == Direction.RIGHT
        dir_down = self.direction == Direction.DOWN
        dir_left = self.direction == Direction.LEFT

        food_up = self.food[1] < head[1]
        food_down = self.food[1] > head[1]
        food_left = self.food[0] < head[0]
        food_right = self.food[0] > head[0]

        # New features: Normalized coordinates
        head_x = head[0] / self.width
        head_y = head[1] / self.height
        tail_x = tail[0] / self.width
        tail_y = tail[1] / self.height

        obs = np.array(
            [
                float(danger_straight),
                float(danger_right),
                float(danger_left),
                float(dir_up),
                float(dir_right),
                float(dir_down),
                float(dir_left),
                float(food_up),
                float(food_down),
                float(food_left),
                float(food_right),
                head_x,
                head_y,
                tail_x,
                tail_y
            ],
            dtype=np.float32,
        )
        return obs

    def _is_collision(self, position, direction):
        """Would moving from 'position' in 'direction' hit wall/body/obstacle?"""
        x, y = position

        if direction == Direction.UP:
            nx, ny = x, y - 1
        elif direction == Direction.DOWN:
            nx, ny = x, y + 1
        elif direction == Direction.LEFT:
            nx, ny = x - 1, y
        elif direction == Direction.RIGHT:
            nx, ny = x + 1, y
        else:
            nx, ny = x, y

        # Wall
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            return True

        # Body
        if (nx, ny) in self.snake:
            return True

        # Obstacle
        if (nx, ny) in self.obstacles:
            return True

        return False

    def _update_direction(self, action):
        """
        RL action -> new direction.
        0: straight, 1: right, 2: left, 3: straight (same as 0).
        """
        directions = [Direction.UP, Direction.RIGHT,
                      Direction.DOWN, Direction.LEFT]
        idx = directions.index(self.direction)

        if action == 0:
            new_idx = idx
        elif action == 1:
            new_idx = (idx + 1) % 4
        elif action == 2:
            new_idx = (idx - 1) % 4
        else:
            new_idx = idx

        self.direction = directions[new_idx]

    def take_action(self, action=None):
        """
        Apply one step of the game.
        """
        if self.game_over:
            return self._get_observation(), 0.0, True, False, {"score": self.score}

        self.steps += 1

        if action is not None:
            self._update_direction(action)

        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        old_dist = abs(head_x - food_x) + abs(head_y - food_y)

        # Move head
        if self.direction == Direction.UP:
            new_head = (head_x, head_y - 1)
        elif self.direction == Direction.DOWN:
            new_head = (head_x, head_y + 1)
        elif self.direction == Direction.LEFT:
            new_head = (head_x - 1, head_y)
        elif self.direction == Direction.RIGHT:
            new_head = (head_x + 1, head_y)
        else:
            new_head = (head_x, head_y)

        reward = 0.0
        terminated = False

        # --- collisions -> true game over --------------------------------
        if (
            new_head[0] < 0
            or new_head[0] >= self.width
            or new_head[1] < 0
            or new_head[1] >= self.height
            or new_head in self.obstacles
            or new_head in self.snake
        ):
            self.game_over = True
            terminated = True
            reward = -10.0
            return self._get_observation(), reward, terminated, False, {
                "score": self.score
            }

        # --- safe: move snake --------------------------------------------
        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            reward += 10.0
            self.food = self._generate_food()
        else:
            self.snake.pop()

        # REWARD SHAPING (Level 3 Fix):
        # Removed the -0.01 time penalty to stop it from suiciding to save time.
        # Added a tiny positive reward for surviving.
        reward += 0.001

        new_dist = abs(new_head[0] - food_x) + abs(new_head[1] - food_y)
        dist_delta = old_dist - new_dist
        reward += 0.005 * dist_delta  # Increased distance incentive slightly

        # --- loop / stuck detection --------------------------------------
        self.last_positions.append(new_head)
        if len(self.last_positions) == self.last_positions.maxlen:
            unique_cells = len(set(self.last_positions))
            if unique_cells < 20:
                self.game_over = True
                terminated = True
                reward -= 5.0
                return self._get_observation(), reward, terminated, False, {
                    "score": self.score
                }

        return self._get_observation(), reward, terminated, False, {
            "score": self.score
        }

    def _init_pygame(self):
        if self.screen is None:
            pygame.init()
            w_px = self.width * self.cell_size
            h_px = self.height * self.cell_size + 40
            self.screen = pygame.display.set_mode((w_px, h_px))
            pygame.display.set_caption("Snake RL")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 20)

    def render(self, mode="human", fps=30):
        if mode != "human":
            return

        self._init_pygame()
        self.screen.fill(self.BLACK)

        # obstacles
        for ox, oy in self.obstacles:
            rect = pygame.Rect(ox * self.cell_size, oy *
                               self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.GREY, rect)
            pygame.draw.rect(self.screen, self.WHITE, rect, 1)

        # snake
        for i, (x, y) in enumerate(self.snake):
            color = self.GREEN if i == 0 else self.DARK_GREEN
            rect = pygame.Rect(x * self.cell_size, y *
                               self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.WHITE, rect, 1)

        # food
        food_rect = pygame.Rect(
            self.food[0] * self.cell_size, self.food[1] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.RED, food_rect)

        # score text
        score_text = self.font.render(f"Score: {self.score}", True, self.WHITE)
        self.screen.blit(score_text, (5, self.height * self.cell_size + 10))

        if self.game_over:
            # Basic Game Over logic (visuals)
            pass

        pygame.display.flip()
        self.clock.tick(fps)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
