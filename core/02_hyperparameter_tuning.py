"""
STEP 4 - HYPERPARAMETER TUNING
GridSearchCV over the Random Forest, scored with GroupKFold so a
district-crop never appears in both training and validation.

With 323 rows an unconstrained forest memorises. The grid is weighted
towards regularisation: shallow trees, larger leaves, fewer features per split.
"""
import pandas as pd, numpy as np, json
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

OUT = '/mnt/user-data/outputs'
df = pd.read_csv(f'{OUT}/farmerhub_yield_features.csv')
df['yield_ratio'] = df.yield_mt_ha / df.national_avg_yield_mt_ha

DROP = ['yield_mt_ha', 'yield_ratio', 'group_id', 'district', 'region_old',
        'rainfall_flagged', 'national_avg_yield_mt_ha', 'rainfall_norm_mm',
        'organic_matter_pct_mid_scaled', 'total_nitrogen_pct_mid_scaled',
        'cation_exchange_capacity_mid_scaled']
X, y, groups = df.drop(columns=DROP), df.yield_ratio, df.group_id
cat = X.select_dtypes(exclude='number').columns.tolist()

pipe = Pipeline([
    ('prep', ColumnTransformer(
        [('cat', OneHotEncoder(handle_unknown='ignore'), cat)], remainder='passthrough')),
    ('model', RandomForestRegressor(random_state=42, n_jobs=-1))])

grid = {
    'model__n_estimators':     [200, 400],
    'model__max_depth':        [3, 5, 8, None],
    'model__min_samples_leaf': [1, 3, 5, 10],
    'model__max_features':     ['sqrt', 0.5, 1.0],
}

gs = GridSearchCV(pipe, grid, cv=GroupKFold(5),
                  scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
gs.fit(X, y, groups=groups)

print('Best parameters:')
for k, v in gs.best_params_.items():
    print(f'  {k.replace("model__","")}: {v}')
print(f'\nBest CV MAE (yield ratio units): {-gs.best_score_:.4f}')

# Show how much tuning actually moved the needle
res = pd.DataFrame(gs.cv_results_)
print(f'Worst configuration in grid:      {-res.mean_test_score.min():.4f}')
print(f'Default-ish (depth None, leaf 1): '
      f'{-res[(res.param_model__max_depth.isna()) & (res.param_model__min_samples_leaf==1)].mean_test_score.max():.4f}')

json.dump({k.replace('model__', ''): v for k, v in gs.best_params_.items()},
          open('/home/claude/best_params.json', 'w'))

print('\nTop 5 configurations:')
top = res.nsmallest(5, 'rank_test_score')[
    ['param_model__max_depth', 'param_model__min_samples_leaf',
     'param_model__max_features', 'param_model__n_estimators', 'mean_test_score']]
top['mean_test_score'] = -top.mean_test_score
print(top.rename(columns=lambda c: c.replace('param_model__', '')).to_string(index=False))
