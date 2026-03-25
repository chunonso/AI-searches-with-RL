import collections
from typing import List, Tuple, Dict, Set

from snake_game import SnakeGame

GridPos = Tuple[int, int]


def reconstruct_path(
    came_from: Dict[GridPos, GridPos],
    current: GridPos,
) -> List[GridPos]:
    """Reconstruct path from start to goal."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def bfs_to_food(game: SnakeGame) -> List[GridPos]:
    """
    Run Breadth-First Search (BFS) on the grid.
    Returns the shortest path (in number of steps) to the food.
    """
    if not game.snake or game.food is None:
        return []

    start: GridPos = game.snake[0]
    goal: GridPos = game.food

    # Define blocked cells
    blocked: Set[GridPos] = set(game.snake[1:])
    if hasattr(game, "obstacles") and game.obstacles:
        blocked |= set(game.obstacles)

    queue = collections.deque([start])
    visited: Set[GridPos] = {start}
    came_from: Dict[GridPos, GridPos] = {}

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    while queue:
        current = queue.popleft()

        if current == goal:
            return reconstruct_path(came_from, current)

        cx, cy = current
        for dx, dy in directions:
            neighbor = (cx + dx, cy + dy)

            # 1. Check Bounds
            if not (0 <= neighbor[0] < game.width and 0 <= neighbor[1] < game.height):
                continue

            # 2. Check Collisions
            if neighbor in blocked:
                continue

            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    return []  # No path found
