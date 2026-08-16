# FarmerHub AI — Yield Prediction Module: Project Log

**Owner:** Chrishelle Wiafe
**Role:** Machine Learning Models · Yield Prediction · Performance Evaluation
**Course:** CS 254 Final Project
**Team:** Louisa-Lois Adjoka (PM/frontend), Kwasi Bekae Ackonor (CNN disease detection),
Daniel Ekpale (search, assistant, backend), Chrishelle Wiafe (ML/yield)

> Running record of decisions, dead ends and results, maintained as the work happens
> so the final report can be written from evidence rather than memory.

---

## Module specification

| Item | Value |
|---|---|
| Task type | Supervised regression |
| Model | Random Forest Regressor (scikit-learn) |
| Target | Crop yield, Mt/Ha |
| Planned inputs | Rainfall, soil quality, fertilizer, crop type, farm size |
| Evaluation metrics | MAE, RMSE |
| Output into system | A predicted-yield figure feeding the shared Farm Health Dashboard |

---

## Phase 1 — Data acquisition

### Sources evaluated

| Source | Verdict |
|---|---|
| MoFA `PRODUCTION_ESTIMATES_2011.xlsx` | Region + district yield, but **single year (2011)** — too little variation |
| FAOSTAT Ghana | National level, single year — **usable only as a sanity check**, not for training |
| HDX / CHIRPS subnational rainfall | Real Ghana rainfall, 1981–2026, but regions identified only by **PCODE** with no name lookup |
| Kaggle `crop_yield.csv` (1M rows) | **Not Ghana data** — regions are "North/South/East/West", crops include cotton and wheat. Retained as *practice data* for building the pipeline |
| Kaggle `crop_yield_dataset.csv` | Same — generic, not Ghana |
| **MoFA SRID *Facts & Figures 2024* (PDF)** | **Adopted.** Yield by district 2021–2023, rainfall by region 2015–2024, soil fertility by region |
| Ghana Meteorological Agency | No open CSV downloads available |

### Dead ends worth recording

- **GMet does not publish downloadable data**, which is why satellite-derived
  (CHIRPS) and published-table sources were used instead.
- **HDX CHIRPS could not be joined.** The file labels regions with codes like `GH02`
  but ships no code-to-name lookup, and the official boundary dataset could not be
  retrieved. Rather than guess a mapping — which would silently mislabel regions
  with no error to reveal it — this source was set aside in favour of the MoFA
  published rainfall table.
- **District-level joins were abandoned.** Ghana's districts were restructured after
  the 2011 data was published (roughly 170 districts then, 216+ by 2018), so
  historical district names no longer map 1:1 onto current ones. Analysis was moved
  to the region level, where the 10 pre-2019 regions are stable.
- **Country-level data was rejected as a training source.** One row per year gives a
  regression model no variation to learn from; it can only memorise. Country figures
  are retained for sanity-checking predictions.

### Datasets built

Transcribed from the MoFA PDF (Tables 2.6, 2.8, 4.7–4.17) into:

- `ghana_district_crop_yield_2021_2023.csv` — 326 rows, 11 crops, 84 districts
- `ghana_rainfall_by_region.csv` — 112 rows, 2015–2024
- `ghana_soil_by_region.csv` — 10 regions, 5 soil properties as min/max/mid
- `farmerhub_yield_training_data.csv` — the three joined into one table

> The four intermediate CSVs above are not shipped in this folder. Only the
> final cleaned merge, `farmerhub_yield_training_data_clean.csv`, is included;
> the intermediates are kept with the transcription working files.

**Join logic:** rainfall for 2021–2022 exists only for the pre-2019 10-region scheme
and for 2023 only for the current 16-region scheme, so a `region_old` column maps
current regions back to their pre-2019 parents. Soil (a 2018 snapshot) joins on the
same column.

---

## Phase 2 — Data cleaning

Cleaning was applied before this folder was assembled; the cleaned result is
`farmerhub_yield_training_data_clean.csv`. Summary:

- **3 yield rows dropped** — implausible values (groundnut at 15.40 Mt/Ha against a
  national potential of 3.50). Target variables cannot be imputed without inventing
  the answer, so removal was the only option.
- **8 rainfall values replaced** with the region's own median. Five fall in 2022 and
  fail in opposite directions, indicating a fault in the published table rather than
  a weather event.
- **One cleaning rule was written, tested and rejected.** Measuring against the PDF's
  printed 30-year averages flagged normal Central-region years, because those printed
  averages contradict the yearly figures they summarise. Replaced with a rule based
  on each region's own observed median.

**Result:** 323 rows, zero missing values, 19.8% carrying an imputed rainfall figure.

---

## Phase 3 — Modelling

### Pipeline established on practice data

Built and validated on the Kaggle practice set before touching real Ghana data, so
that pipeline bugs and data problems could not be confused with one another.

Structure: load → sample → train/test split (80/20) → one-hot encode categorical
columns → fit → evaluate.

### Results on practice data (50,000 rows)

| Model | MAE | RMSE |
|---|---|---|
| Linear Regression | 0.397 | 0.498 |
| Random Forest (200 trees, depth 15) | 0.409 | 0.511 |

Random Forest did **not** beat the linear baseline here. Recorded as-is rather than
tuned until it won: the practice data is synthetic and its underlying relationship is
close to linear, so this is the expected result and reporting it honestly is more
useful than a favourable one.

### Validation checks adopted

1. **Runs without error** — minimum bar
2. **Train vs test gap** — 0.293 vs 0.409 MAE, a small gap indicating no overfitting
3. **Beats a naive baseline** — 70.6% better than always predicting the mean; this is
   the check that demonstrates the model uses its inputs rather than merely fitting
4. **Directional sanity check** — increasing rainfall raised predicted yield,
   consistent with rainfall being the dominant feature

### Feature importance (practice data)

Rainfall 62.7%, fertilizer use ~21%, irrigation ~13%, temperature 1.8%.
Region, soil type and crop did not reach the top ten. Expected to differ on real
Ghana data; to be re-run and compared.

---

---

## Phase 4 — Modelling on the real Ghana data

Practice data retired. All figures below come from
`farmerhub_yield_training_data_clean.csv` (323 rows).

All of this runs in `random_forest_yield.ipynb`, which is the canonical
deliverable for this module.

### 4.1 Feature engineering

Nine features built from the raw columns, each with an agronomic rationale:

| Feature | Rationale |
|---|---|
| `rainfall_ratio` | 1,200 mm is dry for Western but wet for Greater Accra. What matters is rainfall *relative to what that region normally gets*, not the absolute figure |
| `rainfall_anomaly_mm` | Signed departure from the regional norm |
| `rainfall_is_dry_year` / `rainfall_is_wet_year` | Threshold flags at 0.9× and 1.1× the norm |
| `ph_distance_from_optimal` | Most staples prefer pH ≈ 6.25. Distance from that optimum matters more than pH itself, which is not linearly related to yield |
| `n_to_p_ratio` | Nutrient balance — an imbalance limits uptake even when both nutrients are individually adequate (Liebig's law of the minimum) |
| `soil_fertility_index` | Organic matter, nitrogen and CEC, each min-max scaled then averaged, so no unit dominates |
| `rain_x_fertility` | Water and nutrients are only useful together: fertile soil in a drought yields little |
| `crop_yield_headroom` | Potential minus national average — how much room the crop has to improve |

### 4.2 Two methodological problems found, and fixed

**Problem 1 — leakage from random splitting.** The same district-crop appears in
2021, 2022 and 2023 (110 groups across 323 rows). A random train/test split puts
near-duplicate rows on both sides and flatters the result:

| Split method | MAE |
|---|---|
| Random 5-fold | 0.986 |
| **Grouped 5-fold** (district-crop kept whole) | **1.295** |

The random split was overstating performance by roughly 24%. All subsequent
evaluation uses `GroupKFold`.

**Problem 2 — the baseline was too easy.** "Always predict the overall mean" gives
MAE 8.989, which any model beats trivially, because yields differ hugely by crop
(cassava ≈ 35 Mt/Ha, soyabean ≈ 2). The honest baseline is **looking up each crop's
mean yield**, which scores MAE 1.050. Measured against that, the first Random Forest
was *worse than a lookup table* — it was learning crop identity, not agronomy.

**Fix: target reframing.** The model now predicts `yield ÷ crop's national average`
and multiplies back to Mt/Ha, which removes crop scale and forces it to learn what
makes a district out- or under-perform for its own crop.

| Target | MAE (Mt/Ha) |
|---|---|
| Raw yield | 1.293 |
| Yield ratio → Mt/Ha | 1.177 |

### 4.3 Tuning

`GridSearchCV` over 96 configurations, scored with `GroupKFold(5)`.

**Best:** `max_depth=5`, `min_samples_leaf=1`, `max_features=1.0`, `n_estimators=400`

Tuning changed almost nothing — best 0.1585 against 0.1605 for default settings, a
1.2% gain, with the entire grid spanning 0.1585–0.1869. **The binding constraint is
the data, not the hyperparameters.** Reported as found rather than presented as a
tuning success.

### 4.4 Final results

Out-of-fold, `GroupKFold(5)`, in Mt/Ha:

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Tuned Random Forest | 1.096 | 2.189 | **10.89%** |
| Baseline: crop-mean lookup | 1.050 | 1.999 | 12.40% |

**The two metrics disagree, and the reason matters.** On MAE the Random Forest is
4.3% *worse*; on MAPE it is 12% *better*. MAE is an absolute measure, so it is
dominated by the high-tonnage crops — a 4 Mt/Ha error on cassava (mean 35) counts
far more than a 0.25 error on millet (mean 2.4), even though the millet prediction
is proportionally far worse. The per-crop breakdown confirms this: the model beats
the baseline on **6 of 11 crops** — cocoyam, cowpea, groundnut, millet, rice,
soyabean — and loses on the four highest-yield crops.

**Honest conclusion: on 323 rows the model is roughly level with a lookup table,
better proportionally and worse in absolute tonnage.** That is a real finding about
what this dataset can support, not a failure of implementation.

### 4.5 Feature importance

| Feature | Importance |
|---|---|
| Soil organic matter | 0.236 |
| Region: Volta | 0.189 |
| Rainfall × fertility interaction | 0.119 |
| Crop: rice | 0.086 |
| Region: Oti | 0.056 |
| Year | 0.051 |
| Rainfall (mm) | 0.037 |

Soil organic matter is the strongest single predictor, and the engineered
`rain_x_fertility` interaction ranks third — evidence that the feature engineering
contributed real signal rather than noise. Raw rainfall ranks low, which is
consistent with rainfall being regional and therefore near-constant within a region.

---

## Open items

- [ ] Consider per-crop models, or grouping crops by yield scale, to stop the
      high-tonnage crops dominating MAE
- [ ] Quantify the top-ten selection bias — compare predictions against the national
      average yields in Table 4.6
- [ ] Fertilizer input unavailable at regional level; decide whether to drop it from
      the specification or substitute a proxy
- [ ] Farm size also unavailable — the proposal lists it as an input
- [ ] Agree the handoff interface with the team — likely
      `predict_yield(region, crop, rainfall, soil…) → Mt/Ha` for the dashboard

---

## Ethics and fairness notes accumulating for the report

- **Selection bias is the headline issue.** Training only on the top ten districts per
  crop means the model learns what makes a strong farm strong. Deployed to an average
  smallholder it will over-promise, and an over-optimistic yield forecast can lead a
  farmer to over-commit on inputs, credit or sales. This must be disclosed in the app,
  not just the report.
- **Geographic fairness.** Rainfall is regional and soil is a single 2018 snapshot, so
  two farms in the same region receive identical environmental inputs regardless of
  actual local conditions.
- **Transparency.** Random Forest feature importances give a usable explanation of
  which factors drove a prediction, supporting the proposal's commitment that
  recommendations remain explainable.
- **Advisory framing.** Given three years of data, 323 rows and known source errors,
  predictions must be presented as indicative rather than reliable forecasts.
