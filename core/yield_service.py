"""
=============================================================================
FarmerHub AI — Yield Prediction Service
=============================================================================
Backend interface for the yield model.

The farmer supplies REGION and CROP only. Everything else the model needs --
soil chemistry, rainfall, crop potential -- is looked up from reference tables
built at startup, because none of it is farm-specific and a smallholder cannot
be expected to know their soil's cation exchange capacity.

    from yield_service import YieldService

    svc = YieldService()
    svc.options()                          # populate the two dropdowns
    svc.predict('ASHANTI', 'MAIZE')        # -> dict

Requires yield_model.joblib, which is written by section 9 of
random_forest_yield.ipynb. Run the notebook once before using this.

IMPORTANT: predictions are REGIONAL, not farm-specific. Every maize farmer in
Ashanti receives the same figure. UI copy should read "typical for maize in
Ashanti", not "your farm's yield".
=============================================================================
"""
import numpy as np
import pandas as pd
import joblib


class YieldService:

    # Paths are relative to the directory uvicorn is run from (backend
    # project root), matching how main.py loads models/yield_model.joblib.
    MODEL_PATH = 'models/yield_model.joblib'
    DATA_PATH = 'data/farmerhub_yield_training_data_clean.csv'
    MAPE = 0.109      # model's mean absolute percentage error, from notebook §7

    def __init__(self, model_path=None, data_path=None):
        bundle = joblib.load(model_path or self.MODEL_PATH)
        self.model = bundle['model']
        self.features = bundle['features']
        self.national_avg = bundle['national_avg']

        df = pd.read_csv(data_path or self.DATA_PATH)

        # ---- reference tables: what the farmer does not have to supply ----
        # Soil is one 2018 snapshot per region -- 10 rows total.
        soil_cols = ['soil_ph_mid', 'organic_matter_pct_mid',
                     'total_nitrogen_pct_mid', 'avail_phosphorus_mg_kg_mid',
                     'cation_exchange_capacity_mid']
        self._soil = df.groupby('region_old')[soil_cols].first().to_dict('index')

        # Each region's median rainfall, needed to compute rainfall_ratio.
        # A single prediction row cannot compute its own regional median, so
        # it has to come from a table built over the whole dataset.
        self._rain_norm = df.groupby('region_old').rainfall_mm.median().to_dict()

        # Most recent observed rainfall, used when the caller passes none.
        # Keyed on the CURRENT region, not region_old: from 2023 the rainfall
        # tables switched to the 16-region scheme, so Savannah, Northern and
        # North East each have their own figure despite sharing one region_old.
        # Keying this on region_old would collapse them to a single value and
        # silently return the wrong rainfall for two regions out of three.
        self._rain_latest = (df.sort_values('year')
                               .groupby('region').rainfall_mm.last().to_dict())

        self._potential = df.groupby('crop').potential_yield_mt_ha.first().to_dict()

        # The notebook min-max scales these three nutrient measures across the
        # whole dataset before averaging them into soil_fertility_index. A
        # single prediction row cannot reproduce that, so the min and max are
        # captured here and the identical scaling is applied at predict time.
        self._fertility_cols = ['organic_matter_pct_mid', 'total_nitrogen_pct_mid',
                                'cation_exchange_capacity_mid']
        self._fertility_range = {c: (df[c].min(), df[c].max())
                                 for c in self._fertility_cols}

        # Regions were redrawn in 2019; the tables above are keyed on the
        # pre-2019 names, so current names are mapped back.
        self._to_old = df.set_index('region').region_old.to_dict()

        self.regions = sorted(df.region.unique())
        self.crops = sorted(df.crop.unique())

    # -----------------------------------------------------------------
    def options(self):
        """Dropdown contents for the dashboard."""
        return {'regions': self.regions, 'crops': self.crops}

    # -----------------------------------------------------------------
    def predict(self, region, crop, year=2024, rainfall_mm=None):
        """Predict yield for a region-crop pair.

        region      : one of self.regions
        crop        : one of self.crops
        year        : defaults to 2024
        rainfall_mm : optional. Pass a forecast if a weather API is wired in;
                      otherwise the region's latest observed figure is used.
                      Note this moves the prediction very little -- rainfall
                      barely varies within a region in the training data.

        Returns a dict ready to render.
        """
        region = str(region).upper().strip()
        crop = str(crop).upper().strip()

        if crop not in self.national_avg:
            raise ValueError(f'Unknown crop {crop!r}. Options: {self.crops}')
        key = self._to_old.get(region)
        if key is None:
            raise ValueError(f'Unknown region {region!r}. Options: {self.regions}')

        soil = self._soil[key]
        norm = self._rain_norm[key]
        rain = (self._rain_latest[region] if rainfall_mm is None
                else float(rainfall_mm))
        nat = self.national_avg[crop]
        pot = self._potential[crop]

        row = self._build_row(region, crop, year, rain, norm, soil, nat, pot)

        # The model predicts yield RELATIVE to the crop's national average,
        # so the output is multiplied back into Mt/Ha.
        predicted = float(self.model.predict(row)[0] * nat)
        margin = predicted * self.MAPE
        ratio = rain / norm

        return {
            'predicted_yield_mt_ha': round(predicted, 2),
            'range_low_mt_ha': round(predicted - margin, 2),
            'range_high_mt_ha': round(predicted + margin, 2),
            'national_average_mt_ha': round(nat, 2),
            'potential_yield_mt_ha': round(pot, 2),
            'vs_national_average_pct': round(100 * (predicted / nat - 1), 1),
            'rainfall_used_mm': round(rain),
            'rainfall_vs_normal': ('drier than usual' if ratio < 0.9 else
                                   'wetter than usual' if ratio > 1.1 else
                                   'about normal'),
            'region': region,
            'crop': crop,
            'year': year,
            'confidence': 'indicative',
            'basis': 'regional averages, not farm-specific measurements',
            'caveat': ('Trained on the ten best-performing districts per crop, '
                       'so this figure is likely optimistic for an average farm.'),
        }

    # -----------------------------------------------------------------
    def _build_row(self, region, crop, year, rain, norm, soil, nat, pot):
        """Rebuild the engineered features exactly as the notebook does.

        These formulas must stay identical to section 3 of
        random_forest_yield.ipynb. If they drift, the model still returns a
        number -- it is just silently wrong, with no error to warn anyone.
        """
        rainfall_ratio = rain / norm
        ph = soil['soil_ph_mid']
        phos = soil['avail_phosphorus_mg_kg_mid']

        # Min-max scale each nutrient using the ranges captured from the
        # training data, then average -- identical to notebook section 3.
        scaled = []
        for c in self._fertility_cols:
            lo, hi = self._fertility_range[c]
            scaled.append((soil[c] - lo) / (hi - lo) if hi > lo else 0.0)
        fertility = float(np.mean(scaled))

        row = pd.DataFrame([{
            'region': region,
            'crop': crop,
            'year': year,
            'rainfall_mm': rain,
            'rainfall_ratio': rainfall_ratio,
            'rainfall_anomaly_mm': rain - norm,
            'rainfall_is_dry_year': int(rainfall_ratio < 0.9),
            'rainfall_is_wet_year': int(rainfall_ratio > 1.1),
            'soil_ph_mid': ph,
            'ph_distance_from_optimal': abs(ph - 6.25),
            'organic_matter_pct_mid': soil['organic_matter_pct_mid'],
            'total_nitrogen_pct_mid': soil['total_nitrogen_pct_mid'],
            'avail_phosphorus_mg_kg_mid': phos,
            'cation_exchange_capacity_mid': soil['cation_exchange_capacity_mid'],
            'n_to_p_ratio': (soil['total_nitrogen_pct_mid'] / phos) if phos else 0.0,
            'soil_fertility_index': fertility,
            'rain_x_fertility': rainfall_ratio * fertility,
            'potential_yield_mt_ha': pot,
            'crop_yield_headroom': pot - nat,
        }])
        return row[self.features]

    # -----------------------------------------------------------------
    def explain(self, top_n=5):
        """Top features by importance -- for a 'why this prediction' panel."""
        prep = self.model.named_steps['prep']
        ohe = prep.named_transformers_['cat']
        cat = ['region', 'crop']
        names = (list(ohe.get_feature_names_out(cat)) +
                 [f for f in self.features if f not in cat])
        imp = pd.DataFrame({
            'feature': names,
            'importance': self.model.named_steps['model'].feature_importances_})
        return imp.nlargest(top_n, 'importance').to_dict('records')


if __name__ == '__main__':
    svc = YieldService()

    opts = svc.options()
    print(f'{len(opts["regions"])} regions, {len(opts["crops"])} crops')
    print('crops:', ', '.join(opts['crops']))

    print('\n--- maize in Ashanti ---')
    for k, v in svc.predict('ASHANTI', 'MAIZE').items():
        print(f'  {k}: {v}')

    print('\n--- same crop across regions ---')
    for r in ['ASHANTI', 'NORTHERN', 'VOLTA', 'UPPER WEST']:
        out = svc.predict(r, 'MAIZE')
        print(f"  {r:<12} {out['predicted_yield_mt_ha']:>6} Mt/Ha  "
              f"({out['vs_national_average_pct']:+.1f}% vs national)")

    print('\n--- effect of a rainfall forecast ---')
    for mm in [800, 1200, 1600]:
        out = svc.predict('ASHANTI', 'MAIZE', rainfall_mm=mm)
        print(f"  {mm} mm -> {out['predicted_yield_mt_ha']} Mt/Ha "
              f"({out['rainfall_vs_normal']})")

    print('\n--- top features ---')
    for f in svc.explain():
        print(f"  {f['feature']:<28} {f['importance']:.3f}")
