"""
Yield Model Connector for FarmerHub AI
-------------------------------------------
Bridges Chrishelle's yield model into the {(row, col): risk_score}
format that grid_builder.build_risk_weights() and route_planner.py
expect. Deliberately has NO TensorFlow dependency -- anything that only
needs yield predictions shouldn't be forced to install/import TF just
because disease_connector.py lives in the same project.
"""

import joblib

from .route_planner import yield_to_risk


def load_yield_model(model_path='yield_model.joblib'):
    """Loads the saved bundle: {'model', 'features', 'national_avg'}."""
    bundle = joblib.load(model_path)
    return bundle['model'], bundle['national_avg']


def get_yield_risk_map(plots_data, model, national_avg):
    """
    plots_data: dict (row, col) -> dict with the fields predict_yield()
                needs per plot:
                  region, crop, year, rainfall_mm, soil_ph, organic_matter,
                  nitrogen, phosphorus, cec, region_median_rainfall,
                  potential_yield
                This has to come from farm registration data / soil
                sensors / weather API -- it does not exist yet in what
                we've built so far.

    Returns: dict (row, col) -> risk score 0-1
    """
    # imported here (not top-level) so this file doesn't hard-require
    # yield_model.py to be present just to be READ
    from .yield_model import predict_yield

    risk = {}
    for plot, d in plots_data.items():
        predicted = predict_yield(
            model, national_avg,
            region=d['region'], crop=d['crop'], year=d['year'],
            rainfall_mm=d['rainfall_mm'], soil_ph=d['soil_ph'],
            organic_matter=d['organic_matter'], nitrogen=d['nitrogen'],
            phosphorus=d['phosphorus'], cec=d['cec'],
            region_median_rainfall=d['region_median_rainfall'],
            potential_yield=d['potential_yield'],
        )
        # potential_yield is the attainable ceiling for THIS plot's
        # conditions, so it's a better risk reference than the flat
        # national average -- a plot with low potential shouldn't be
        # judged against a number it could never realistically hit.
        risk[plot] = yield_to_risk(predicted, max_expected_yield=d['potential_yield'])
    return risk
