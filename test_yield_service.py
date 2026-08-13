"""
Verifies that YieldService reproduces the notebook's predictions.

The wrapper rebuilds the engineered features from lookup tables instead of
from the full dataset. If those formulas drift from section 3 of
random_forest_yield.ipynb, the model still returns a number -- it is just
silently wrong, with nothing to signal the error. This test catches that.

Run:  python test_yield_service.py
"""
import numpy as np
import pandas as pd
import joblib

from yield_service import YieldService

TOLERANCE = 0.011      # predictions are rounded to 2dp


def notebook_features(df):
    """Feature engineering copied verbatim from notebook section 3."""
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


def main():
    df = pd.read_csv('farmerhub_yield_training_data_clean.csv')
    bundle = joblib.load('yield_model.joblib')
    svc = YieldService()

    d = notebook_features(df)
    expected = bundle['model'].predict(d[bundle['features']]) * d.national_avg_yield_mt_ha

    # 2023 rows: the service defaults to each region's latest rainfall,
    # which is the 2023 figure, so the two paths should agree exactly.
    test = d[d.year == 2023].drop_duplicates(['region', 'crop'])

    diffs, worst = [], None
    for i, row in test.iterrows():
        got = svc.predict(row.region, row.crop, year=2023)['predicted_yield_mt_ha']
        diff = abs(got - expected[i])
        diffs.append(diff)
        if worst is None or diff > worst[0]:
            worst = (diff, row.region, row.crop, got, expected[i])

    diffs = np.array(diffs)
    print(f'compared {len(diffs)} region-crop pairs')
    print(f'max difference:  {diffs.max():.4f} Mt/Ha')
    print(f'mean difference: {diffs.mean():.4f} Mt/Ha')
    print(f'worst case: {worst[1]} / {worst[2]} — '
          f'service {worst[3]:.3f} vs notebook {worst[4]:.3f}')

    assert diffs.max() < TOLERANCE, (
        f'\nFAIL: service diverges from the notebook by up to {diffs.max():.4f} Mt/Ha.\n'
        'The feature formulas in yield_service._build_row have drifted from\n'
        'notebook section 3, or a lookup table is keyed on the wrong column.')

    # Interface sanity checks
    opts = svc.options()
    assert len(opts['regions']) == 14 and len(opts['crops']) == 11
    out = svc.predict('ASHANTI', 'MAIZE')
    assert out['range_low_mt_ha'] < out['predicted_yield_mt_ha'] < out['range_high_mt_ha']
    for bad in [('NOWHERE', 'MAIZE'), ('ASHANTI', 'BANANA')]:
        try:
            svc.predict(*bad)
            raise AssertionError(f'expected ValueError for {bad}')
        except ValueError:
            pass

    print('\nPASS — service matches the notebook and the interface behaves.')


if __name__ == '__main__':
    main()
