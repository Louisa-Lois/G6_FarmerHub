"""
Farm Grid Builder for FarmerHub AI
------------------------------------
Converts farm layout + your teammates' model outputs into the
`grid` and `risk_weights` structures that astar_route.py expects.
"""


def build_grid(rows, cols, obstacles=None):
    """
    rows, cols: farm dimensions in plots (e.g. 10x10 grid of plots)
    obstacles: list of (row, col) tuples that are NOT walkable
               (sheds, water tanks, fences, etc.)

    Returns: dict mapping (row, col) -> True (walkable) / False (blocked)
    """
    obstacles = set(obstacles or [])
    grid = {}
    for r in range(rows):
        for c in range(cols):
            grid[(r, c)] = (r, c) not in obstacles
    return grid


def build_risk_weights(yield_scores, disease_scores, yield_weight=0.5, disease_weight=0.5):
    """
    yield_scores: dict (row, col) -> predicted risk from the yield model,
                  0.0 (healthy/high yield) to 1.0 (severe/low yield).
                  NOTE: you'll need to invert Chrishelle's raw yield
                  output into a "risk" score first, e.g.
                  risk = 1 - (predicted_yield / max_expected_yield)

    disease_scores: dict (row, col) -> CNN confidence that the plot has
                     disease/pest damage, 0.0 to 1.0. Only plots with an
                     uploaded photo will have an entry -- default 0 elsewhere.

    yield_weight, disease_weight: how much each source contributes to
                     the final risk score. Must sum to 1.0.

    Returns: dict (row, col) -> combined risk score, 0.0 to 1.0
    """
    assert abs(yield_weight + disease_weight - 1.0) < 1e-6, "weights must sum to 1.0"

    all_plots = set(yield_scores) | set(disease_scores)
    risk_weights = {}
    for plot in all_plots:
        y = yield_scores.get(plot, 0.0)
        d = disease_scores.get(plot, 0.0)
        risk_weights[plot] = min(1.0, yield_weight * y + disease_weight * d)
    return risk_weights


if __name__ == "__main__":
    # Example: a 4x4 farm, obstacle at (1,1) (e.g. a water tank)
    grid = build_grid(rows=4, cols=4, obstacles=[(1, 1)])

    # Example model outputs (normally these come from Chrishelle's and
    # Kwasi's modules at runtime)
    yield_risk = {(2, 2): 0.9, (3, 1): 0.4}       # low predicted yield -> high risk
    disease_risk = {(2, 2): 0.6, (0, 3): 0.7}      # CNN flagged these plots

    risk_weights = build_risk_weights(yield_risk, disease_risk)

    print("Grid:", grid)
    print("Risk weights:", risk_weights)
