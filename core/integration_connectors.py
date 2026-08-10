"""
Combined Model Integration for FarmerHub AI
------------------------------------------------
Thin combiner over yield_connector.py and disease_connector.py.

Import from the specific connector file (yield_connector / disease_connector)
if you only need one -- that avoids requiring TensorFlow just to use the
yield side. This file exists for the case where you genuinely need both
at once.
"""

from .yield_connector import load_yield_model, get_yield_risk_map
from .disease_connector import load_disease_model, get_disease_risk_map


def build_combined_risk_weights(plots_data, plot_photos,
                                 yield_model_path='yield_model.joblib',
                                 disease_model_path='farmerhub_disease_model.keras',
                                 class_names_path='class_names.json',
                                 yield_weight=0.4, disease_weight=0.6):
    from .grid_builder import build_risk_weights

    yield_model, national_avg = load_yield_model(yield_model_path)
    yield_risk = get_yield_risk_map(plots_data, yield_model, national_avg)

    disease_model, class_names = load_disease_model(disease_model_path, class_names_path)
    disease_risk = get_disease_risk_map(plot_photos, disease_model, class_names)

    return build_risk_weights(yield_risk, disease_risk, yield_weight, disease_weight)
