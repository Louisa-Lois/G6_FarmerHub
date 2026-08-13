"""
Tests for astar_route.py

Run with: pytest backend/tests/test_astar.py -v
(or just: python -m pytest tests/test_astar.py -v  from inside backend/)
"""

import heapq
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.astar_route import astar


# ---------------------------------------------------------------------
# Ground truth: a plain Dijkstra implementation, independent of A*'s
# heuristic entirely. If A* is admissible, its returned cost must always
# equal Dijkstra's -- Dijkstra is guaranteed optimal by construction
# (it has no heuristic to get wrong), so it's a trustworthy baseline to
# test A* against.
# ---------------------------------------------------------------------

def dijkstra(grid, start, goal, risk_weights):
    def neighbors(node):
        r, c = node
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nxt = (r + dr, c + dc)
            if grid.get(nxt, False):
                yield nxt

    def move_cost(node):
        base = 1.0
        risk = risk_weights.get(node, 0.0)
        return max(0.1, base - risk)

    dist = {start: 0.0}
    heap = [(0.0, start)]
    visited = set()

    while heap:
        d, current = heapq.heappop(heap)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return d
        for nxt in neighbors(current):
            nd = d + move_cost(nxt)
            if nd < dist.get(nxt, float('inf')):
                dist[nxt] = nd
                heapq.heappush(heap, (nd, nxt))

    return float('inf')


# ---------------------------------------------------------------------
# Sanity tests
# ---------------------------------------------------------------------

def test_finds_valid_path_simple_grid():
    """Basic case: open 4x4 grid, no obstacles, no risk. Confirms A*
    finds SOME valid path and it actually connects start to goal."""
    grid = {(r, c): True for r in range(4) for c in range(4)}
    path, cost = astar(grid, start=(0, 0), goal=(3, 3), risk_weights={})

    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (3, 3)
    # every consecutive pair in the path must be adjacent (no teleporting)
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1
    assert cost == 6.0  # shortest path on an open grid = 6 steps * 1.0 cost


def test_no_path_when_goal_unreachable():
    """Goal completely walled off -> A* must report no path, not crash
    or return a bogus route."""
    grid = {(r, c): True for r in range(4) for c in range(4)}
    # wall off (3,3) on all sides
    grid[(2, 3)] = False
    grid[(3, 2)] = False
    path, cost = astar(grid, start=(0, 0), goal=(3, 3), risk_weights={})
    assert path is None
    assert cost == float('inf')


def test_obstacle_forces_detour():
    """A* must route AROUND an obstacle, not attempt to pass through it."""
    grid = {(r, c): True for r in range(3) for c in range(3)}
    grid[(1, 1)] = False  # block the direct diagonal-ish shortcut
    path, cost = astar(grid, start=(0, 0), goal=(2, 2), risk_weights={})
    assert path is not None
    assert (1, 1) not in path


# ---------------------------------------------------------------------
# Adversarial test: this is the one that catches the heuristic bug.
# Before the fix (heuristic assumed cost-per-step=1.0 while move_cost
# could drop to 0.1 for high-risk cells), A* could return a route up to
# ~282% more expensive than optimal, because an overestimating heuristic
# lets A* prune away the actually-cheapest path too early.
# ---------------------------------------------------------------------

def test_matches_dijkstra_on_adversarial_risk_grid():
    """A high-risk 'shortcut' corridor sits off the straight-line path.
    A broken (overestimating) heuristic can cause A* to ignore it and
    take the naive straight path instead, even though routing through
    the discounted corridor is actually cheaper. A* must match Dijkstra
    exactly here."""
    rows, cols = 8, 8
    grid = {(r, c): True for r in range(rows) for c in range(cols)}

    # a corridor of heavily-discounted (high-risk) cells, off the direct path
    risk_weights = {}
    for r in range(rows):
        risk_weights[(r, 5)] = 0.9  # move_cost here = 0.1, big discount

    start, goal = (0, 0), (7, 7)

    astar_path, astar_cost = astar(grid, start, goal, risk_weights)
    dijkstra_cost = dijkstra(grid, start, goal, risk_weights)

    assert astar_path is not None
    # the core correctness property: A*'s cost must equal true optimal cost
    assert math.isclose(astar_cost, dijkstra_cost, rel_tol=1e-9), (
        f"A* returned cost {astar_cost}, but true optimal (Dijkstra) is "
        f"{dijkstra_cost} -- heuristic is not admissible"
    )


def test_matches_dijkstra_on_multiple_random_grids():
    """Broader confidence check: run several randomized risk-weighted
    grids and confirm A* always matches Dijkstra's optimal cost, not
    just the one hand-crafted adversarial case above."""
    import random
    random.seed(7)

    for trial in range(10):
        rows, cols = 6, 6
        grid = {(r, c): True for r in range(rows) for c in range(cols)}
        # randomly block a few cells, but never start/goal
        obstacles = random.sample(
            [(r, c) for r in range(rows) for c in range(cols)
             if (r, c) not in [(0, 0), (rows - 1, cols - 1)]],
            k=5,
        )
        for o in obstacles:
            grid[o] = False

        risk_weights = {
            (r, c): random.choice([0.0, 0.3, 0.6, 0.9])
            for r in range(rows) for c in range(cols)
        }

        start, goal = (0, 0), (rows - 1, cols - 1)
        astar_path, astar_cost = astar(grid, start, goal, risk_weights)
        dijkstra_cost = dijkstra(grid, start, goal, risk_weights)

        if dijkstra_cost == float('inf'):
            assert astar_path is None
        else:
            assert math.isclose(astar_cost, dijkstra_cost, rel_tol=1e-9), (
                f"Trial {trial}: A* cost {astar_cost} != Dijkstra cost {dijkstra_cost}"
            )


if __name__ == "__main__":
    # allow running without pytest installed, for a quick manual check
    tests = [
        test_finds_valid_path_simple_grid,
        test_no_path_when_goal_unreachable,
        test_obstacle_forces_detour,
        test_matches_dijkstra_on_adversarial_risk_grid,
        test_matches_dijkstra_on_multiple_random_grids,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
