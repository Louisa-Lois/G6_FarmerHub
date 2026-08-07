"""
Model Integration Connectors for FarmerHub AI
-------------------------------------------------
Bridges Chrishelle's yield model and Kwasi's disease model into the
{(row, col): risk_score} format that grid_builder.build_risk_weights()
and route_planner.py expect.
"""

import json

import joblib
import numpy as np
import tensorflow as tf

from .route_planner import yield_to_risk


# ---------------------------------------------------------------------
# Chrishelle's yield model
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Kwasi's disease model
# ---------------------------------------------------------------------

def load_disease_model(model_path='farmerhub_disease_model.keras',
                        class_names_path='class_names.json'):
    """
    Loads the CNN and its class-name mapping.

    IMPORTANT: class_names.json does not exist yet. Kwasi's notebook
    currently never saves train_ds.class_names anywhere -- it only
    exists inside his Colab session. Ask him to add:
        json.dump(class_names, open('class_names.json', 'w'))
    right after training, and send you that file along with the .keras
    model. Without it there's no way to know which output index maps
    to which disease.
    """
    model = tf.keras.models.load_model(model_path)
    with open(class_names_path) as f:
        class_names = json.load(f)
    return model, class_names


def get_disease_risk_map(plot_photos, model, class_names, img_size=(256, 256)):
    """
    plot_photos: dict (row, col) -> image file path. Only plots with an
                 uploaded leaf photo will have an entry here -- plots
                 without one simply don't appear in the returned dict
                 (same convention build_risk_weights already expects).
    class_names: list of class labels, index-aligned with the model's
                 output layer (see load_disease_model note above).

    Returns: dict (row, col) -> disease risk score 0-1. A predicted
             "healthy" class maps to 0 risk; any disease class maps to
             the model's confidence in that prediction.
    """
    risk = {}
    for plot, path in plot_photos.items():
        img = tf.keras.utils.load_img(path, target_size=img_size)
        arr = tf.keras.utils.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        top_class = class_names[top_idx]
        confidence = float(preds[top_idx])

        risk[plot] = 0.0 if 'healthy' in top_class.lower() else confidence
    return risk


# ---------------------------------------------------------------------
# Full combined pipeline (once real inputs exist)
# ---------------------------------------------------------------------

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
