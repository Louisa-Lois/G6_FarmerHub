"""
Verifies that YieldService reproduces the trained model's predictions.

The wrapper rebuilds the engineered features from lookup tables instead of
from the full dataset. If those formulas drift from core/yield_model.py's
engineer_features(), the model still returns a number -- it is just
silently wrong, with nothing to signal the error. This test catches that.

Run from the backend project root:  pytest tests/test_yield_service.py
"""
import numpy as np
import pandas as pd
import joblib
import pytest

from core.yield_service import YieldService

TOLERANCE = 0.011      # predictions are rounded to 2dp


def notebook_features(df):
    """Feature engineering copied verbatim from core/yield_model.py."""
    d = df.copy()
    norm = d.groupby('region_old').rainfall_mm.transform('median')
    d['rainfall_ratio'] = d.rainfall_mm / norm
    d['rainfall_anomaly_mm'] = d.rainfall_mm - norm
    d['rainfall_is_dry_year'] = (d.rainfall_ratio < 0.9).astype(int)
    d['rainfall_is_wet_year'] = (d.rainfall_ratio > 1.1).astype(int)
    d['ph_distance_from_optimal'] = (d.soil_ph_mid - 6.25).abs()
    d['n_to_p_ratio'] = (d.total_nitrogen_pct_mid /
                         d.avail_phosphorus_mg_kg_mid.replace(0, np.nan))
    d['n_to_p_ratio'] = d.n_to_p_ratio.fillna(d.n_to_p_ratio.median())
    parts = []
    for c in ['organic_matter_pct_mid', 'total_nitrogen_pct_mid',
              'cation_exchange_capacity_mid']:
        r = d[c].max() - d[c].min()
        parts.append((d[c] - d[c].min()) / r if r else d[c] * 0)
    d['soil_fertility_index'] = np.mean(parts, axis=0)
    d['rain_x_fertility'] = d.rainfall_ratio * d.soil_fertility_index
    d['crop_yield_headroom'] = (d.potential_yield_mt_ha -
                                d.national_avg_yield_mt_ha)
    return d


@pytest.fixture(scope="module")
def svc():
    return YieldService()


def test_matches_trained_model(svc):
    df = pd.read_csv('data/farmerhub_yield_training_data_clean.csv')
    bundle = joblib.load('models/yield_model.joblib')

    d = notebook_features(df)
    expected = bundle['model'].predict(d[bundle['features']]) * d.national_avg_yield_mt_ha

    # 2023 rows: the service defaults to each region's latest rainfall,
    # which is the 2023 figure, so the two paths should agree exactly.
    test = d[d.year == 2023].drop_duplicates(['region', 'crop'])

    diffs = []
    for i, row in test.iterrows():
        got = svc.predict(row.region, row.crop, year=2023)['predicted_yield_mt_ha']
        diffs.append(abs(got - expected[i]))

    diffs = np.array(diffs)
    assert diffs.max() < TOLERANCE, (
        f'service diverges from the trained model by up to {diffs.max():.4f} Mt/Ha'
    )


def test_options(svc):
    opts = svc.options()
    assert len(opts['regions']) == 14
    assert len(opts['crops']) == 11


def test_prediction_range_is_sane(svc):
    out = svc.predict('ASHANTI', 'MAIZE')
    assert out['range_low_mt_ha'] < out['predicted_yield_mt_ha'] < out['range_high_mt_ha']
    assert out['potential_yield_mt_ha'] > 0


@pytest.mark.parametrize("region,crop", [('NOWHERE', 'MAIZE'), ('ASHANTI', 'BANANA')])
def test_unknown_inputs_raise(svc, region, crop):
    with pytest.raises(ValueError):
        svc.predict(region, crop)
