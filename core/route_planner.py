"""
Route Planner for FarmerHub AI
---------------------------------
Ties astar_route.py and grid_builder.py together into the actual
module output: an ordered inspection/irrigation route + estimated
travel time, built from teammates' model outputs.
"""

from astar_route import astar
from grid_builder import build_grid, build_risk_weights


# ---------------------------------------------------------------------
# 1. Yield -> risk conversion (integration hook for Chrishelle's model)
# ---------------------------------------------------------------------

def yield_to_risk(predicted_yield, max_expected_yield):
    """
    Converts a raw predicted-yield value (tonnes/hectare, higher = better)
    into a 0-1 risk score (higher = worse), for use in build_risk_weights.

    predicted_yield: output of the yield regression model for one plot
    max_expected_yield: the yield a healthy plot of this crop/region is
                         expected to hit -- e.g. a regional average or the
                         max value seen in training data. Team needs to
                         agree on this reference number per crop type.
    """
    if max_expected_yield <= 0:
        return 0.0
    risk = 1.0 - (predicted_yield / max_expected_yield)
    return max(0.0, min(1.0, risk))  # clamp to [0, 1]


def yield_scores_from_predictions(predictions, max_expected_yield):
    """
    predictions: dict (row, col) -> predicted_yield from Chrishelle's model
    Returns: dict (row, col) -> risk score, ready for build_risk_weights
    """
    return {
        plot: yield_to_risk(pred, max_expected_yield)
        for plot, pred in predictions.items()
    }


# ---------------------------------------------------------------------
# 2. Multi-stop routing
# ---------------------------------------------------------------------
# Plain astar() only goes point A to point B. A real inspection route
# needs to visit SEVERAL high-risk plots in one trip. True optimal
# multi-stop ordering is the Traveling Salesman Problem (NP-hard) --
# overkill for this project. We use a greedy nearest-neighbor approach:
# from the current position, always go to the closest unvisited priority
# plot next. It won't be perfectly optimal, but it's fast and reasonable
# for a farm with a handful of flagged plots per day.

def plan_route(grid, start, priority_plots, risk_weights):
    """
    grid: from build_grid()
    start: (row, col) where the farmer begins (e.g. farmhouse or gate)
    priority_plots: list of (row, col) plots that must be visited
                     (e.g. every plot with risk_weights above some threshold)
    risk_weights: from build_risk_weights()

    Returns: dict with 'route' (ordered list of (row, col) waypoints,
             including every step of the walked path) and 'stops'
             (just the priority plots in visit order).
    """
    remaining = list(priority_plots)
    current = start
    full_path = [start]
    stop_order = []

    while remaining:
        # find the nearest remaining plot by actual A* path cost
        best_plot, best_path, best_cost = None, None, float('inf')
        for plot in remaining:
            path, cost = astar(grid, current, plot, risk_weights)
            if path is not None and cost < best_cost:
                best_plot, best_path, best_cost = plot, path, cost

        if best_plot is None:
            # remaining plots are unreachable (e.g. sealed off by obstacles)
            break

        full_path.extend(best_path[1:])  # skip duplicate of current node
        stop_order.append(best_plot)
        current = best_plot
        remaining.remove(best_plot)

    return {"route": full_path, "stops": stop_order}


# ---------------------------------------------------------------------
# 3. Travel time estimation
# ---------------------------------------------------------------------

def estimate_travel_time(route, plot_size_meters, walking_speed_m_per_min):
    """
    route: list of (row, col) waypoints, e.g. plan_route(...)["route"]
    plot_size_meters: real-world distance between adjacent grid cells
    walking_speed_m_per_min: assumed walking pace, e.g. 60 m/min (~3.6 km/h)

    Returns: estimated time in minutes (float)
    """
    steps = len(route) - 1
    if steps <= 0:
        return 0.0
    distance_meters = steps * plot_size_meters
    return distance_meters / walking_speed_m_per_min


# ---------------------------------------------------------------------
# 4. Full pipeline example (integration hooks in one place)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import random

    # --- simulate a bigger, more realistic farm (20x20 plots) ---
    ROWS, COLS = 20, 20
    random.seed(42)
    obstacles = [(r, c) for r in range(ROWS) for c in range(COLS) if random.random() < 0.05]
    grid = build_grid(ROWS, COLS, obstacles=obstacles)

    # --- simulate Chrishelle's yield model output ---
    raw_yield_predictions = {
        (r, c): random.uniform(0.5, 4.0)
        for r in range(ROWS) for c in range(COLS)
        if random.random() < 0.3  # only some plots have fresh predictions
    }
    yield_risk = yield_scores_from_predictions(raw_yield_predictions, max_expected_yield=4.0)

    # --- simulate Kwasi's CNN disease output (only plots with uploaded photos) ---
    disease_risk = {
        (r, c): random.uniform(0.0, 1.0)
        for r in range(ROWS) for c in range(COLS)
        if random.random() < 0.15
    }

    risk_weights = build_risk_weights(yield_risk, disease_risk, yield_weight=0.4, disease_weight=0.6)

    # --- pick priority plots: anything above a risk threshold ---
    THRESHOLD = 0.6
    priority_plots = [p for p, r in risk_weights.items() if r >= THRESHOLD and grid.get(p, False)]
    print(f"Flagged {len(priority_plots)} high-risk plots out of {ROWS*COLS} total")

    start = (0, 0)
    result = plan_route(grid, start, priority_plots, risk_weights)

    travel_time = estimate_travel_time(result["route"], plot_size_meters=8.0, walking_speed_m_per_min=60.0)

    print("Visit order:", result["stops"])
    print(f"Total walked steps: {len(result['route']) - 1}")
    print(f"Estimated travel time: {travel_time:.1f} minutes")
