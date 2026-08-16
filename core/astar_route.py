"""
A* Route Optimization for FarmerHub AI
----------------------------------------
Finds the lowest-cost inspection/irrigation route across a farm grid,
where "cost" combines travel distance and risk-priority from the
yield prediction and disease detection modules.

This uses a generic grid representation for now -- swap in your real
plot coordinates / risk scores once the grid design is finalized.
"""

import heapq
import math


def astar(grid, start, goal, risk_weights):
    """
    grid: dict mapping (row, col) -> True if walkable, False if obstacle
    start, goal: (row, col) tuples
    risk_weights: dict mapping (row, col) -> risk score (0.0 = no risk,
                  higher = more urgent to visit). Visiting a high-risk
                  node should be CHEAPER so A* prefers routing through it.

    Returns: (path, total_cost) where path is a list of (row, col) nodes,
             or (None, float('inf')) if no path exists.
    """

    def heuristic(node):
        # Straight-line distance to goal -- never overestimates true
        # travel distance, so it stays admissible.
        (r1, c1), (r2, c2) = node, goal
        return 0.01 * math.hypot(r2 - r1, c2 - c1)

    def neighbors(node):
        r, c = node
        # 4-directional movement; add diagonals here if your grid allows it
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nxt = (r + dr, c + dc)
            if grid.get(nxt, False):  # only walkable, non-obstacle cells
                yield nxt

    def move_cost(node):
        base = 1.0
        risk = risk_weights.get(node, 0.0)
        # Apply a much steeper discount (0.01 instead of 0.1) to force A*
        # to aggressively detour towards high-risk plots.
        return max(0.01, base - (risk * 2))

    open_heap = [(heuristic(start), 0.0, start, None)]
    best_g = {start: 0.0}
    came_from = {}

    while open_heap:
        f, g, current, parent = heapq.heappop(open_heap)

        if current in came_from and g > best_g[current]:
            continue  # stale heap entry, skip

        came_from[current] = parent

        if current == goal:
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path, g

        for nxt in neighbors(current):
            tentative_g = g + move_cost(nxt)
            if tentative_g < best_g.get(nxt, float("inf")):
                best_g[nxt] = tentative_g
                heapq.heappush(
                    open_heap, (tentative_g + heuristic(nxt), tentative_g, nxt, current)
                )

    return None, float("inf")


if __name__ == "__main__":
    # Tiny 4x4 demo grid: all walkable except one obstacle at (1,1)
    grid = {(r, c): True for r in range(4) for c in range(4)}
    grid[(1, 1)] = False

    # Risk scores from disease/yield modules -- higher = more urgent
    risk_weights = {(2, 2): 0.8, (3, 1): 0.5}

    path, cost = astar(grid, start=(0, 0), goal=(3, 3), risk_weights=risk_weights)
    print("Path:", path)
    print("Total cost:", round(cost, 2))
