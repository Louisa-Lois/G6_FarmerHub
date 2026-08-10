"""
=============================================================================
FarmerHub AI — Yield Prediction Module
Random Forest Regressor
=============================================================================
Module owner : Chrishelle Wiafe
Role         : Machine Learning Models, Yield Prediction, Performance Evaluation
Course       : CS 254 Final Project

Input  : farmerhub_yield_training_data_clean.csv  (323 rows)
Output : trained model + MAE/RMSE evaluation + predict_yield() for the dashboard

=============================================================================
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH  = 'farmerhub_yield_training_data_clean.csv'
MODEL_PATH = 'yield_model.joblib'
RANDOM_STATE = 42

# Found by GridSearchCV over 96 configurations, scored with GroupKFold(5).
# See 02_hyperparameter_tuning.py and PROJECT_LOG.md section 4.3.
BEST_PARAMS = dict(n_estimators=400, max_depth=5,
                   min_samples_leaf=1, max_features=1.0)


# =============================================================================
# 1. FEATURE ENGINEERING
# =============================================================================
def engineer_features(df):
    """Build yield-weather-soil features. Each has an agronomic rationale --
    raw columns alone carry little signal."""
    df = df.copy()

    # --- Weather -------------------------------------------------------
    # 1,200 mm is dry for Western but wet for Greater Accra, so absolute
    # rainfall is not comparable across regions. What matters is rainfall
    # relative to what that region normally receives.
    norm = df.groupby('region_old').rainfall_mm.transform('median')
    df['rainfall_ratio']      = df.rainfall_mm / norm
    df['rainfall_anomaly_mm'] = df.rainfall_mm - norm
    df['rainfall_is_dry_year'] = (df.rainfall_ratio < 0.9).astype(int)
    df['rainfall_is_wet_year'] = (df.rainfall_ratio > 1.1).astype(int)

    # --- Soil ----------------------------------------------------------
    # Most staples prefer pH near 6.25. Distance from that optimum matters
    # more than pH itself, which is not linearly related to yield.
    df['ph_distance_from_optimal'] = (df.soil_ph_mid - 6.25).abs()

    # Nutrient balance: an imbalance limits uptake even when both nutrients
    # are individually adequate (Liebig's law of the minimum).
    df['n_to_p_ratio'] = (df.total_nitrogen_pct_mid /
                          df.avail_phosphorus_mg_kg_mid.replace(0, np.nan))
    df['n_to_p_ratio'] = df.n_to_p_ratio.fillna(df.n_to_p_ratio.median())

    # Composite fertility: three nutrient measures, each min-max scaled so
    # no single unit dominates, then averaged.
    parts = []
    for c in ['organic_matter_pct_mid', 'total_nitrogen_pct_mid',
              'cation_exchange_capacity_mid']:
        rng = df[c].max() - df[c].min()
        parts.append((df[c] - df[c].min()) / rng if rng else df[c] * 0)
    df['soil_fertility_index'] = np.mean(parts, axis=0)

    # --- Interaction ---------------------------------------------------
    # Water and nutrients are only useful together: fertile soil in a
    # drought yields little, heavy rain on poor soil leaches what there is.
    df['rain_x_fertility'] = df.rainfall_ratio * df.soil_fertility_index

    # --- Crop context --------------------------------------------------
    df['crop_yield_headroom'] = (df.potential_yield_mt_ha -
                                 df.national_avg_yield_mt_ha)

    # The same district-crop appears in 2021, 2022 and 2023. Those rows are
    # near-duplicates, so they must be kept together when splitting.
    df['group_id'] = df.district + '__' + df.crop
    return df


FEATURES = [
    'region', 'crop', 'year',
    'rainfall_mm', 'rainfall_ratio', 'rainfall_anomaly_mm',
    'rainfall_is_dry_year', 'rainfall_is_wet_year',
    'soil_ph_mid', 'ph_distance_from_optimal',
    'organic_matter_pct_mid', 'total_nitrogen_pct_mid',
    'avail_phosphorus_mg_kg_mid', 'cation_exchange_capacity_mid',
    'n_to_p_ratio', 'soil_fertility_index', 'rain_x_fertility',
    'potential_yield_mt_ha', 'crop_yield_headroom',
]


def build_model():
    """Random Forest inside a pipeline. One-hot encoding is inside the
    pipeline so the encoder is fitted on training folds only."""
    cat = ['region', 'crop']
    return Pipeline([
        ('prep', ColumnTransformer(
            [('cat', OneHotEncoder(handle_unknown='ignore'), cat)],
            remainder='passthrough')),
        ('model', RandomForestRegressor(random_state=RANDOM_STATE,
                                        n_jobs=-1, **BEST_PARAMS)),
    ])


# =============================================================================
# 2. EVALUATION
# =============================================================================
def evaluate(df):
    """Out-of-fold evaluation under GroupKFold.

    A district-crop never appears in both training and validation. A plain
    random split leaks the 2021/2022/2023 near-duplicates across the divide
    and overstates performance by roughly 24%.

    The model predicts yield RELATIVE to the crop's national average, then
    multiplies back to Mt/Ha. Predicting raw yield lets crop scale dominate
    (cassava ~35 Mt/Ha vs soyabean ~2), so the model learns crop identity
    instead of agronomy.
    """
    X, groups = df[FEATURES], df.group_id
    ratio = df.yield_mt_ha / df.national_avg_yield_mt_ha

    pred = np.zeros(len(df))
    base = np.zeros(len(df))

    for tr, te in GroupKFold(n_splits=5).split(X, ratio, groups):
        m = build_model().fit(X.iloc[tr], ratio.iloc[tr])
        pred[te] = m.predict(X.iloc[te]) * df.national_avg_yield_mt_ha.iloc[te]

        # Baseline: look up the crop's mean yield, computed on training
        # folds only. This is the bar the model must clear -- "predict the
        # overall mean" is trivially beatable because crops differ so much.
        cm = df.iloc[tr].groupby('crop').yield_mt_ha.mean()
        base[te] = (df.iloc[te].crop.map(cm)
                    .fillna(df.iloc[tr].yield_mt_ha.mean()).values)

    actual = df.yield_mt_ha.values

    def metrics(p):
        return (np.abs(actual - p).mean(),
                np.sqrt(((actual - p) ** 2).mean()),
                (np.abs(actual - p) / actual).mean() * 100)

    m_mae, m_rmse, m_mape = metrics(pred)
    b_mae, b_rmse, b_mape = metrics(base)

    print('=' * 62)
    print('EVALUATION — out-of-fold, GroupKFold(5), units Mt/Ha')
    print('=' * 62)
    print(f'{"":<26}{"MAE":>8}{"RMSE":>9}{"MAPE":>9}')
    print(f'{"Random Forest":<26}{m_mae:8.3f}{m_rmse:9.3f}{m_mape:8.2f}%')
    print(f'{"Baseline: crop mean":<26}{b_mae:8.3f}{b_rmse:9.3f}{b_mape:8.2f}%')
    print(f'\nvs baseline:  MAE {100*(1-m_mae/b_mae):+.1f}%   '
          f'MAPE {100*(1-m_mape/b_mape):+.1f}%')
    print('\nThe two metrics disagree because MAE is absolute and so is')
    print('dominated by high-tonnage crops. A 4 Mt/Ha miss on cassava')
    print('(mean 35) outweighs a 0.25 miss on millet (mean 2.4), even')
    print('though the millet prediction is proportionally far worse.')

    per = (df.assign(pred=pred, base=base)
             .groupby('crop')
             .apply(lambda g: pd.Series({
                 'n': len(g),
                 'mean_yield': g.yield_mt_ha.mean(),
                 'rf_mae': np.abs(g.yield_mt_ha - g.pred).mean(),
                 'base_mae': np.abs(g.yield_mt_ha - g.base).mean()}),
                 include_groups=False))
    per['rf_better'] = np.where(per.rf_mae < per.base_mae, 'yes', 'no')
    print('\n--- PER-CROP MAE ---')
    print(per.round(3).to_string())
    print(f"\nModel beats the baseline on {(per.rf_better=='yes').sum()} "
          f"of {len(per)} crops.")

    return pd.DataFrame({'region': df.region, 'district': df.district,
                         'crop': df.crop, 'year': df.year,
                         'actual_mt_ha': actual,
                         'predicted_mt_ha': pred.round(3),
                         'baseline_mt_ha': base.round(3)})


def show_importance(model, df):
    ohe = model.named_steps['prep'].named_transformers_['cat']
    names = (list(ohe.get_feature_names_out(['region', 'crop'])) +
             [f for f in FEATURES if f not in ('region', 'crop')])
    imp = (pd.DataFrame({'feature': names,
                         'importance': model.named_steps['model'].feature_importances_})
           .sort_values('importance', ascending=False))
    print('\n--- TOP 10 FEATURES ---')
    print(imp.head(10).round(4).to_string(index=False))
    return imp


# =============================================================================
# 3. PREDICTION INTERFACE — what the dashboard calls
# =============================================================================
def predict_yield(model, national_avg, region, crop, year, rainfall_mm,
                  soil_ph, organic_matter, nitrogen, phosphorus, cec,
                  region_median_rainfall, potential_yield):
    """Predict yield in Mt/Ha for one farm. Returns a float."""
    row = pd.DataFrame([{
        'region_old': region, 'region': region, 'crop': crop, 'year': year,
        'rainfall_mm': rainfall_mm, 'soil_ph_mid': soil_ph,
        'organic_matter_pct_mid': organic_matter,
        'total_nitrogen_pct_mid': nitrogen,
        'avail_phosphorus_mg_kg_mid': phosphorus,
        'cation_exchange_capacity_mid': cec,
        'potential_yield_mt_ha': potential_yield,
        'national_avg_yield_mt_ha': national_avg[crop],
        'district': 'query', 'yield_mt_ha': np.nan,
    }])
    # rainfall_ratio needs the regional norm, which a single row cannot
    # compute -- it is passed in from the reference table instead.
    row['rainfall_ratio'] = rainfall_mm / region_median_rainfall
    row['rainfall_anomaly_mm'] = rainfall_mm - region_median_rainfall
    row['rainfall_is_dry_year'] = int(row.rainfall_ratio.iloc[0] < 0.9)
    row['rainfall_is_wet_year'] = int(row.rainfall_ratio.iloc[0] > 1.1)
    row['ph_distance_from_optimal'] = abs(soil_ph - 6.25)
    row['n_to_p_ratio'] = nitrogen / phosphorus if phosphorus else 0.0
    row['soil_fertility_index'] = np.mean([organic_matter, nitrogen, cec])
    row['rain_x_fertility'] = (row.rainfall_ratio.iloc[0] *
                               row.soil_fertility_index.iloc[0])
    row['crop_yield_headroom'] = potential_yield - national_avg[crop]

    ratio = model.predict(row[FEATURES])[0]
    return float(ratio * national_avg[crop])


# =============================================================================
# 4. MAIN
# =============================================================================
if __name__ == '__main__':
    raw = pd.read_csv(DATA_PATH)
    df = engineer_features(raw)
    print(f'Loaded {len(df)} rows, {df.group_id.nunique()} district-crop groups\n')

    predictions = evaluate(df)

    # Final model, fitted on all available data, for deployment
    final = build_model().fit(df[FEATURES],
                             df.yield_mt_ha / df.national_avg_yield_mt_ha)
    importance = show_importance(final, df)

    national_avg = df.groupby('crop').national_avg_yield_mt_ha.first().to_dict()
    joblib.dump({'model': final, 'features': FEATURES,
                 'national_avg': national_avg}, MODEL_PATH)

    predictions.to_csv('yield_model_predictions.csv', index=False)
    importance.to_csv('yield_model_feature_importance.csv', index=False)
    print(f'\nSaved: {MODEL_PATH}, yield_model_predictions.csv, '
          f'yield_model_feature_importance.csv')

    # Worked example of the dashboard call
    demo = predict_yield(
        final, national_avg, region='ASHANTI', crop='MAIZE', year=2024,
        rainfall_mm=1300, soil_ph=5.7, organic_matter=6.915,
        nitrogen=0.26, phosphorus=3.49, cec=4.81,
        region_median_rainfall=1239, potential_yield=5.5)
    print(f'\nExample — maize, Ashanti, 1300 mm rainfall: {demo:.2f} Mt/Ha')
