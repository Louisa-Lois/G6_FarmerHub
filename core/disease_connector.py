"""
Disease Model Connector for FarmerHub AI
----------------------------------------------
Bridges Kwasi's disease model into the {(row, col): risk_score} format
that grid_builder.build_risk_weights() and route_planner.py expect.
Split out from yield_connector.py so TensorFlow is only imported when
disease detection is actually needed.
"""

import json

import numpy as np
import tensorflow as tf


def load_disease_model(model_path='farmerhub_disease_model.keras',
                        class_names_path='class_names.json'):
    """
    Loads the CNN and its class-name mapping.

    IMPORTANT: class_names.json must be saved alongside the .keras model
    -- model.save() does not preserve the output-index-to-label mapping
    on its own.
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
