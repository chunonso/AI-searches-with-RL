# Compare.py
import pygame
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from stable_baselines3 import PPO

from snake_env import SnakeEnv
from snake_game import SnakeGame, Direction

# --- IMPORTS FROM YOUR ALGORITHM FILES ---
from Astar import a_star_to_food
from Bfs import bfs_to_food

pygame.init()

FONT_BIG = pygame.font.SysFont("arial", 21, bold=True)
FONT_MED = pygame.font.SysFont("arial", 19, bold=True)
FONT_SMALL = pygame.font.SysFont("arial", 14)


def direction_from_points(a, b) -> Optional[Direction]:
    """Return the Direction needed to move from cell a -> b (adjacent)."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay

    if dx == 0 and dy == -1:
        return Direction.UP
    if dx == 1 and dy == 0:
        return Direction.RIGHT
    if dx == 0 and dy == 1:
        return Direction.DOWN
    if dx == -1 and dy == 0:
        return Direction.LEFT
    return None


def draw_snake_board(surface: pygame.Surface, game_like, cell_size: int):
    surface.fill((5, 5, 5))
    width = game_like.width
    height = game_like.height

    # Grid
    for x in range(width + 1):
        pygame.draw.line(surface, (35, 35, 35), (x * cell_size, 0),
                         (x * cell_size, height * cell_size))
    for y in range(height + 1):
        pygame.draw.line(surface, (35, 35, 35), (0, y * cell_size),
                         (width * cell_size, y * cell_size))

    # Snake
    for i, (sx, sy) in enumerate(game_like.snake):
        rect = pygame.Rect(sx * cell_size, sy * cell_size,
                           cell_size, cell_size)
        color = (0, 220, 0) if i == 0 else (0, 180, 0)
        pygame.draw.rect(surface, color, rect)
        pygame.draw.rect(surface, (10, 60, 10), rect, 1)

    # Food
    if getattr(game_like, "food", None) is not None:
        fx, fy = game_like.food
        rect = pygame.Rect(fx * cell_size, fy * cell_size,
                           cell_size, cell_size)
        pygame.draw.rect(surface, (210, 40, 40), rect)
        pygame.draw.rect(surface, (100, 0, 0), rect, 1)

    # Obstacles
    obstacles = getattr(game_like, "obstacles", [])
    for ox, oy in obstacles:
        rect = pygame.Rect(ox * cell_size, oy * cell_size,
                           cell_size, cell_size)
        pygame.draw.rect(surface, (120, 120, 120), rect)
        pygame.draw.rect(surface, (60, 60, 60), rect, 1)


@dataclass
class Panel:
    name: str
    strategy: str
    rect: pygame.Rect
    cell_size: int
    rl_env: Optional[SnakeEnv] = None
    rl_obs: Optional[object] = None
    game: Optional[SnakeGame] = None
    done: bool = False
    score: int = 0
    steps: int = 0
    death_time: float = 0.0  # survival time in seconds for this episode
    # total decision time (ms) spent in this episode
    plan_time_ms_total: float = 0.0


@dataclass
class AgentStats:
    name: str
    scores: List[int] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    times: List[float] = field(
        default_factory=list)           # survival seconds
    # avg planning ms per step, per episode
    plan_ms: List[float] = field(default_factory=list)
    # 1 if score>=1 else 0
    wins: List[int] = field(default_factory=list)

    @property
    def games(self) -> int:
        return len(self.scores)

    @property
    def avg_score(self) -> float:
        return sum(self.scores) / self.games if self.games else 0.0

    @property
    def best_score(self) -> int:
        return max(self.scores) if self.games else 0

    @property
    def avg_steps(self) -> float:
        return sum(self.steps) / self.games if self.games else 0.0

    @property
    def avg_time(self) -> float:
        return sum(self.times) / self.games if self.games else 0.0

    @property
    def win_rate(self) -> float:
        return (sum(self.wins) / self.games) if self.games else 0.0

    @property
    def avg_plan_ms(self) -> float:
        return sum(self.plan_ms) / self.games if self.games else 0.0

    @property
    def last_score(self) -> int:
        return self.scores[-1] if self.games else 0

    def add_episode(self, panel: Panel):
        self.scores.append(int(panel.score))
        self.steps.append(int(panel.steps))
        self.times.append(float(panel.death_time))
        self.wins.append(1 if panel.score >= 1 else 0)

        # average planning ms per step for this episode (avoid division by zero)
        if panel.steps > 0:
            self.plan_ms.append(panel.plan_time_ms_total / panel.steps)
        else:
            self.plan_ms.append(0.0)


def clone_game(base: SnakeGame) -> SnakeGame:
    game = SnakeGame(width=base.width, height=base.height,
                     cell_size=base.cell_size)
    game.snake = list(base.snake)
    game.direction = base.direction
    game.food = base.food
    if hasattr(base, "obstacles"):
        game.obstacles = list(base.obstacles)
    game.score = 0
    game.game_over = False
    return game


def create_episode_panels(panel_rects, cell_size):
    # RL Env (this reset creates the "base" episode state we clone for A* and BFS)
    rl_env = SnakeEnv(render_mode=None)
    obs, info = rl_env.reset()
    base_game = getattr(rl_env, "game", None)

    rl_panel = Panel(
        "RL (PPO)", "rl", panel_rects[0], cell_size,
        rl_env=rl_env, rl_obs=obs, game=base_game
    )
    astar_panel = Panel("A* Search", "astar",
                        panel_rects[1], cell_size, game=clone_game(base_game))
    bfs_panel = Panel("BFS Search", "bfs",
                      panel_rects[2], cell_size, game=clone_game(base_game))

    return [rl_panel, astar_panel, bfs_panel]


def step_panel(panel: Panel, model: PPO, elapsed: float, max_steps: int = 500):
    if panel.done:
        return

    if panel.strategy == "rl":
        t0 = time.perf_counter()
        action, _ = model.predict(panel.rl_obs, deterministic=True)
        panel.plan_time_ms_total += (time.perf_counter() - t0) * 1000.0

        obs, reward, terminated, truncated, info = panel.rl_env.step(action)
        panel.rl_obs = obs
        panel.steps += 1
        panel.score = int(info.get("score", panel.score))

        if terminated or truncated or panel.steps >= max_steps:
            panel.done = True
            if panel.death_time == 0.0:
                panel.death_time = elapsed

    else:
        # SEARCH STRATEGIES (A* / BFS)
        game = panel.game
        if game.game_over or panel.steps >= max_steps:
            panel.done = True
            if panel.death_time == 0.0:
                panel.death_time = elapsed
            return

        # plan path and time it
        t0 = time.perf_counter()
        if panel.strategy == "astar":
            path = a_star_to_food(game)
        else:
            path = bfs_to_food(game)
        panel.plan_time_ms_total += (time.perf_counter() - t0) * 1000.0

        # follow the next step of the planned path (if any)
        if len(path) >= 2:
            head = path[0]
            nxt = path[1]
            new_dir = direction_from_points(head, nxt)
            if new_dir is not None:
                game.direction = new_dir

        _, reward, terminated, truncated, info = game.take_action()
        panel.steps += 1
        panel.score = getattr(game, "score", panel.score)

        if terminated or truncated:
            panel.done = True
            if panel.death_time == 0.0:
                panel.death_time = elapsed


def draw_panel(surface, panel, episode):
    x, y, w, h = panel.rect
    # Card background
    pygame.draw.rect(surface, (25, 25, 25), (x - 4, y -
                     4, w + 8, h + 8), border_radius=12)
    pygame.draw.rect(surface, (80, 80, 80), (x - 4, y - 4,
                     w + 8, h + 8), 2, border_radius=12)

    label_h = 70
    board_rect = pygame.Rect(x, y + label_h, w, h - label_h)
    draw_snake_board(surface.subsurface(board_rect),
                     panel.game, panel.cell_size)

    # Text
    pygame.draw.rect(surface, (35, 35, 35),
                     (x, y, w, label_h), border_radius=12)
    surface.blit(FONT_BIG.render(panel.name, True,
                 (255, 255, 255)), (x + 10, y + 10))

    time_str = f"{panel.death_time:4.1f}s" if panel.death_time > 0 else "--"
    avg_plan = (panel.plan_time_ms_total /
                panel.steps) if panel.steps > 0 else 0.0
    stats = f"Score: {panel.score}  Steps: {panel.steps}  Time: {time_str}  Plan: {avg_plan:4.1f}ms"
    surface.blit(FONT_SMALL.render(
        stats, True, (200, 200, 200)), (x + 10, y + 45))


def draw_scoreboard(surface, stats: Dict[str, AgentStats], rect, episode):
    x, y, w, h = rect
    pygame.draw.rect(surface, (18, 18, 18), rect, border_radius=12)
    pygame.draw.rect(surface, (80, 80, 80), rect, 2, border_radius=12)

    surface.blit(FONT_MED.render(
        f"Episode {episode}", True, (255, 215, 0)), (x + 15, y + 12))

    # Headers
    header1 = "AvgScore | Best | Win% | AvgSteps | AvgTime | AvgPlan"
    surface.blit(FONT_SMALL.render(
        header1, True, (180, 180, 180)), (x + 15, y + 42))

    line_y = y + 62
    for key in ["rl", "astar", "bfs"]:
        s = stats[key]
        txt = (
            f"{s.name:<3}  "
            f"{s.avg_score:7.2f} | "
            f"{s.best_score:4d} | "
            f"{(s.win_rate*100):5.1f}% | "
            f"{s.avg_steps:7.1f} | "
            f"{s.avg_time:7.1f}s | "
            f"{s.avg_plan_ms:7.2f}ms"
        )
        surface.blit(FONT_SMALL.render(
            txt, True, (220, 220, 220)), (x + 15, line_y))
        line_y += 22

    # Small note / legend
    note = "Win% = episodes with score >= 1. Plan = decision-making time per step."
    surface.blit(FONT_SMALL.render(
        note, True, (140, 140, 140)), (x + 15, y + h - 28))


def main():
    width_cells, height_cells, cell_size = 20, 15, 20
    board_px_w = width_cells * cell_size
    board_px_h = height_cells * cell_size + 70

    # Window layout
    window_w = 3 * board_px_w + 360
    window_h = board_px_h + 60
    screen = pygame.display.set_mode((window_w, window_h))
    pygame.display.set_caption("Snake Arena: RL vs A* vs BFS")
    clock = pygame.time.Clock()

    print("[Arena] Loading Model 'snake_model'.")
    try:
        model = PPO.load("snake_model")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("DID YOU RETRAIN? The input shape has changed. Run train_snake.py first!")
        return

    # Panel Rects
    panel_rects = []
    for i in range(3):
        panel_rects.append(pygame.Rect(
            10 + i * (board_px_w + 10), 50, board_px_w, board_px_h))

    stats: Dict[str, AgentStats] = {
        "rl": AgentStats("RL"),
        "astar": AgentStats("A*"),
        "bfs": AgentStats("BFS"),
    }

    panels = None
    episode = 0
    running = True

    while running:
        # start a new episode if none, or the current episode is fully done
        if panels is None or all(p.done for p in panels):
            if panels:
                # record episode results for each strategy
                for p in panels:
                    stats[p.strategy].add_episode(p)

                # Pause to see result
                pygame.display.flip()
                pygame.time.wait(800)

                # cleanup RL env
                for p in panels:
                    if p.rl_env:
                        p.rl_env.close()

            episode += 1
            panels = create_episode_panels(panel_rects, cell_size)
            start_time = time.perf_counter()

        elapsed = time.perf_counter() - start_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # advance each panel by one step
        for p in panels:
            step_panel(p, model, elapsed)

        # draw
        screen.fill((15, 15, 18))
        for p in panels:
            draw_panel(screen, p, episode)

        draw_scoreboard(
            screen,
            stats,
            pygame.Rect(3 * (board_px_w + 10) + 10, 50, 340, 210),
            episode,
        )

        pygame.display.flip()
        clock.tick(15)

    pygame.quit()


if __name__ == "__main__":
    main()
