# FarmerHub AI — Backend & A* Route Optimization

**Module owner:** Daniel — Search Algorithms, Backend, AI Assistant (stretch)

This is the backend server for FarmerHub AI: it hosts every team member's
model behind one FastAPI app, and implements the A\* route optimization
module (farm inspection/irrigation route planning, prioritizing
high-risk plots).

## What's in this module

- **A\* route optimization** (`core/astar_route.py`, `core/grid_builder.py`,
  `core/route_planner.py`) — finds the lowest-cost inspection route across
  a farm grid, using an admissible heuristic and risk-weighted node costs
  from the yield and disease models.
- **Backend integration** (`main.py`) — the single point every other
  module's work passes through: yield prediction, disease detection,
  route planning, and weather advice, each exposed as an HTTP endpoint.

## Project structure

```
backend/
├── main.py                  # FastAPI app -- all endpoints
├── requirements.txt          # pinned versions
├── core/
│   ├── astar_route.py        # A* search
│   ├── grid_builder.py       # farm grid + risk-weight construction
│   ├── route_planner.py      # multi-stop routing + travel time
│   ├── yield_connector.py    # yield model integration (no TF dependency)
│   ├── disease_connector.py  # disease model integration
│   ├── integration_connectors.py  # combines yield + disease
│   ├── yield_model.py        # Chrishelle's yield model
│   └── weather.py            # Louisa-Lois's weather module
├── models/
│   ├── yield_model.joblib
│   ├── disease_model.keras
│   └── class_names.json
└── tests/
    └── test_astar.py         # sanity + adversarial tests vs. Dijkstra
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### Weather endpoint API key

`/weather-advice` needs a real OpenWeatherMap key:

1. Copy `.env.example` to `.env` in this folder.
2. Replace `your_key_here` with a real key.
3. Never commit `.env` or share a real key outside your own file — it's
   already in `.gitignore`.

## Running it

```bash
uvicorn main:app --reload
```

Wait for `Uvicorn running on http://127.0.0.1:8000` (TensorFlow + model
loading takes ~15-20s after that). Open `http://127.0.0.1:8000/docs`
for the interactive API page.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Confirms the server and both models loaded |
| POST | `/predict-yield` | Chrishelle's yield model |
| POST | `/detect-disease` | Kwasi's disease model (file upload) |
| POST | `/plan-route` | A\* route optimization |
| POST | `/weather-advice` | Louisa-Lois's weather-based decision support |

## Testing

```bash
python -m pytest tests/test_astar.py -v
```

`test_astar.py` is the evaluation artifact for the A\* module (same role
Kwasi's classification report and Chrishelle's MAE/MAPE numbers play for
theirs). It includes:
- Sanity checks (valid path found, obstacles avoided, unreachable goals
  correctly reported)
- An adversarial test and a 10-trial randomized comparison against a
  from-scratch Dijkstra implementation, confirming A\*'s heuristic is
  admissible (returns truly optimal routes, not just "a route")

## Known open items

- `DISEASE_IMG_SIZE` in `main.py` is still `(96, 96)` — this matches the
  current CPU-trained prototype model. Once Kwasi's real 256×256 model
  is swapped into `models/`, this constant must change to `(256, 256)`
  in the same commit, or predictions will be silently wrong.
- CORS in `main.py` currently assumes a separate React/Vite dev server.
  If the frontend ends up served directly by FastAPI as static files
  instead, this config becomes unnecessary and should be removed —
  decision pending with Louisa-Lois.
- `demo-frontend/` (if present locally) is a personal dev/testing tool,
  not a project deliverable — it's git-ignored and shouldn't be part of
  the branch that merges into `main`.
# FarmerHub AI 🌱 — Frontend & Integration Branch
**Branch Owner:** Louisa-Lois Adjoka (Project Manager, Frontend & Integration)
**Repository:** [Group-6-Final-Project-FarmerHub](https://github.com/Louisa-Lois/Group-6-Final-Project-FarmerHub.git)

This branch contains the user interface, system integration logic, and weather-based decision support modules for **FarmerHub AI**, an intelligent decision-support system for smallholder farmers in Ghana[cite: 3]. 

While the core machine learning models (Yield Prediction, CNN Disease Detection, and $A^*$ Routing) are developed in parallel branches, this branch serves as the central hub that consolidates their outputs into a single, actionable dashboard[cite: 3].

---

##  Branch Deliverables & Features

### 1. Farm Health Dashboard (Core Integration)
The primary deliverable of this branch is the unified interface that allows farmers to monitor their plots. 
* Aggregates the outputs of the standalone AI modules (Yield, Disease, Routing)[cite: 3].
* Computes and displays an overarching Farm Health Score[cite: 3].
* Provides real-time, color-coded urgency alerts for individual plots requiring immediate attention[cite: 3].

### 2. Weather-Based Decision Support (Stretch Goal)
Successfully implemented the project's weather integration stretch goal[cite: 3].
* Integrates live forecast data via the OpenWeatherMap API[cite: 3].
* Generates rule-based actionable alerts (e.g., "Rainfall expected in the next 48 hours. Delay chemical spraying.") based on local environmental conditions[cite: 3].

### 3. Frontend Architecture
* **Responsive UI:** Built using standard HTML5, Vanilla JavaScript, and Tailwind CSS.
* **Component Design:** Features cascading plot registration forms, a drag-and-drop image upload interface for the CNN model, and a dynamic 2D visual grid to animate the $A^*$ route optimization[cite: 3].
* **API Readiness:** Fully wired with JavaScript `fetch()` logic designed to seamlessly connect to the FastAPI backend via a static file mount upon final merge.

---

##  How to Run This Branch Locally

Because this branch isolates the frontend UI and does not contain the Python backend (`main.py`) or the heavy machine learning models, it can be tested instantly without a Python environment.

### 1. Clone the Repository
```bash
git clone [https://github.com/Louisa-Lois/Group-6-Final-Project-FarmerHub.git](https://github.com/Louisa-Lois/Group-6-Final-Project-FarmerHub.git)
cd Group-6-Final-Project-FarmerHub
git checkout Louisa-Lois
# FarmerHub AI - Yield Prediction

Predicts crop yield in tonnes per hectare for Ghanaian districts, using a
Random Forest regressor trained on Ministry of Food and Agriculture data.

CS 254 final project · module owner **Chrishelle Wiafe** (ML models, yield
prediction, performance evaluation)

---

## Quick start

```bash
pip install pandas numpy scikit-learn joblib
jupyter notebook random_forest_yield.ipynb
```

The notebook runs top to bottom and already has its outputs saved, so it can be
read without executing. Keep all files in the same folder — paths are relative.

### The module

| File | |
|---|---|
| `random_forest_yield.ipynb` | **The deliverable.** Features → tuning → training → evaluation, inline. Outputs saved, so it can be read without running |
| `farmerhub_yield_training_data_clean.csv` | The only data file the notebook loads. 323 rows × 15 columns |
| `02_hyperparameter_tuning.py` | Standalone grid search (~2 min). The same search also runs inline in notebook §6 |
| `PROJECT_LOG.md` | Decisions, rejected data sources, dead ends |

### Backend integration

| File | |
|---|---|
| `yield_service.py` | What the dashboard calls. Farmer supplies **region and crop only**; soil chemistry, rainfall and crop potential are looked up internally |
| `test_yield_service.py` | Verifies the service reproduces the notebook's predictions. Run after any change to either |

```python
from yield_service import YieldService

svc = YieldService()
svc.options()                      # {'regions': [...], 'crops': [...]}
svc.predict('ASHANTI', 'MAIZE')    # -> dict with prediction, range, caveats
svc.explain()                      # top features, for a "why" panel
```

`yield_service.py` needs `yield_model.joblib`, which the notebook writes in
section 9 - run the notebook once first.

The farmer is never asked for soil pH, nitrogen, phosphorus or organic matter.
None of those are farm-specific: soil is one 2018 survey value per region and
rainfall is one value per region-year, so choosing a region and a crop
determines every other input. **This also means predictions are regional, not
personal** - every maize farmer in Ashanti gets the same figure, and UI copy
should say *"typical for maize in Ashanti"* rather than *"your farm's yield"*.

### Retained for reference

Kept in the branch as a record of how the module developed. **Not part of the
merged work** - the notebook above supersedes them.

| File | |
|---|---|
| `yield_model.py` | Earlier script version of the module, before conversion to a notebook. Its hyperparameters are hard-coded from an earlier tuning run, so its numbers differ slightly from the notebook. Its `predict_yield()` function was written for a prediction service that was later dropped and is not used by anything |
| `yield_model.joblib` | Trained model, written by the notebook's section 9 |
| `yield_model_predictions.csv` | Actual vs predicted vs baseline for all 323 rows, written by section 9 |
| `yield_model_feature_importance.csv` | Full feature ranking, written by section 9 |

The last three are regenerated every time the notebook runs, so they will change
on re-execution.

---

## Results

Out-of-fold under `GroupKFold(5)`, in Mt/Ha:

| | MAE | RMSE | MAPE |
|---|---|---|---|
| **Tuned Random Forest** | 1.10 | 2.14 | **10.9%** |
| Baseline — crop-mean lookup | 1.05 | 2.00 | 12.4% |

The two metrics disagree, and that is the finding. MAE is absolute, so
high-tonnage crops dominate it: a 4 Mt/Ha miss on cassava (mean 35) counts more
than a 0.25 miss on millet (mean 2.4), though the millet prediction is
proportionally worse. MAPE weights every crop equally.

Per crop, the model beats the baseline on **6 of 11** - all of them low-tonnage -
and loses on the four largest.

> **Summary:** on 323 rows the model performs about level with a crop-average
> lookup. Better proportionally, worse in absolute tonnage.

**What the model leans on:** region (Volta 0.18) and soil organic matter (0.16)
lead, with the engineered rainfall × fertility interaction fourth (0.06). Raw
rainfall ranks low, which follows from rainfall being recorded per region and so
barely varying within one. Exact ranks shift slightly between runs - see the note
on tuning below.

---

## Data

MoFA SRID, *Agriculture in Ghana: Facts & Figures 2024* (34th ed.),
Tables 2.6, 2.8 and 4.7–4.17.

**323 rows · 11 crops · 84 districts · 14 regions · 2021–2023**

Three tables were merged into the single CSV here:

| Source | Role | Join |
|---|---|---|
| Yield by district | Target variable | base table |
| Rainfall by region | Feature | region + year |
| Soil chemistry by region | Feature | region only (2018 snapshot) |

Ghana redrew its regions in 2019, from 10 to 16. Rainfall for 2021–2022 is
published under the old scheme and 2023 under the new, so a `region_old` column
maps current regions to their pre-2019 parents and the merge runs in two halves.

Cleaning dropped 3 implausible yield rows - one reported groundnut at 15.40
Mt/Ha against a national *potential* of 3.50 - and replaced 8 rainfall values.
Yield is the target, so corrupt values there were removed rather than imputed:
imputing a target means inventing the answer the model is meant to learn. Full
detail in `PROJECT_LOG.md` §2.

---

## Method

Nine features are engineered from the raw columns. The reasoning matters more
than the list:

- **Rainfall is made relative.** 1,200 mm is a drought in Western region and a
  flood in Greater Accra, so absolute millimetres are not comparable. The model
  sees rainfall as a ratio to the regional norm, where 1.0 is a normal year.
- **pH becomes distance from optimum.** Most staples prefer pH ≈ 6.25, and both
  extremes hurt yield. Raw pH asks a tree to learn a U-shape; distance from the
  optimum makes it a straight line.
- **Soil nutrients are rescaled before combining.** Organic matter runs 2–7%
  while nitrogen runs 0.01–0.26%, so averaging them raw would let organic matter
  dominate on units alone.
- **Rainfall × fertility is an interaction.** Water and nutrients only help
  together - fertile soil in a drought yields little. This feature ranks third
  in importance, so it carried real signal.

### Two decisions that cost accuracy on paper

Both cases where the obvious approach gave a *better-looking* number and a
*worse* model.

**Grouped splitting, not random.** The same district-crop appears in 2021, 2022
and 2023 - 110 groups across 323 rows - and those rows are near-duplicates. A
random split scores MAE 0.99; keeping groups whole scores 1.30. The random split
was overstating performance by about 24%.

**A ratio target, not raw yield.** Predicting raw yield across crops spanning 2
to 35 Mt/Ha meant the model mostly learned crop identity, and did so *worse* than
a lookup table. It now predicts yield ÷ the crop's national average and converts
back, which forces it to learn what makes a district out- or under-perform for
its own crop.

**On tuning:** the 96-configuration grid spans only ~0.03 in MAE, and the top
configurations sit within 0.0002 of each other - close enough that the winner
changes between runs on numerical noise. There is no meaningful best setting to
find here. The constraint is the data, not the hyperparameters.

---

## Limitations

Properties of the available data rather than implementation faults. These belong
in the dashboard UI, not only in the report.

**Selection bias is the main one.** MoFA publishes only the ten best-performing
districts per crop, so every training row is a high performer. The model
over-predicts for average farms - precisely the smallholders FarmerHub exists to
serve.

**Predictions are regional, not personal.** Soil is one value per region and
rainfall one per region-year, so every maize farmer in Ashanti gets the same
number. The dashboard should read *"typical for maize in Ashanti"*, not *"your
farm's yield"*.

**Rainfall barely moves the output**, since it hardly varies within a region
here. Wiring in a weather forecast will not change predictions much.

**Fertilizer and farm size are missing.** Both are listed as model inputs in the
project proposal, but MoFA publishes only national fertilizer prices and import
volumes - no regional application rates.

**Three years, 323 rows.** Too small to support strong claims.

Predictions are advisory. An over-optimistic forecast can lead a farmer to
over-commit on inputs, credit or sales.

---

## Next

- Per-crop models, so cassava and yam stop dominating the MAE
- Quantify the selection bias against the Table 4.6 national averages
- Decide whether to drop fertilizer from the specification or find a proxy
