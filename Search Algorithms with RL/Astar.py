import heapq
from typing import List, Tuple, Dict, Set

from snake_game import SnakeGame

GridPos = Tuple[int, int]


def manhattan(a: GridPos, b: GridPos) -> int:
    """Heuristic: Manhattan distance (taxicab geometry) between two grid cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(
    came_from: Dict[GridPos, GridPos],
    current: GridPos,
) -> List[GridPos]:
    """Reconstruct path from start to current (goal) using the parent map."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def a_star_to_food(game: SnakeGame) -> List[GridPos]:
    """
    Run A* search on the current SnakeGame grid:
    - Start: snake head
    - Goal:  food position
    - Blocked: snake body (except head) and obstacles

    Returns:
        List of grid cells [start, step1, step2, ..., goal].
        Returns empty list [] if no path is found.
    """
    if not game.snake or game.food is None:
        return []

    start: GridPos = game.snake[0]
    goal: GridPos = game.food

    # Define blocked cells (snake body + obstacles)
    # We allow the tail to be a valid move spot strictly speaking,
    # but for safety, we usually treat the whole body as blocked.
    blocked: Set[GridPos] = set(game.snake[1:])

    if hasattr(game, "obstacles") and game.obstacles:
        blocked |= set(game.obstacles)

    # Priority Queue for A*: stores (f_score, g_score, (x, y))
    open_set = []
    heapq.heappush(open_set, (manhattan(start, goal), 0, start))

    came_from: Dict[GridPos, GridPos] = {}

    # Cost from start to current node
    g_score: Dict[GridPos, int] = {start: 0}

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right

    while open_set:
        # Pop the node with the lowest f_score
        _, g, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        # Optimization: If we found a shorter way to this node already, skip
        if g > g_score.get(current, float('inf')):
            continue

        cx, cy = current
        for dx, dy in directions:
            neighbor = (cx + dx, cy + dy)

            # 1. Check Bounds
            if not (0 <= neighbor[0] < game.width and 0 <= neighbor[1] < game.height):
                continue

            # 2. Check Collisions (Walls/Body/Obstacles)
            if neighbor in blocked:
                continue

            tentative_g = g + 1

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + manhattan(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))

    return []
