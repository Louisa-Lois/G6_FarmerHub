# FarmerHub AI

An intelligent decision-support system for smallholder farmers in Ghana.

FarmerHub combines three AI techniques behind a single dashboard: a **Random
Forest** regressor that predicts crop yield, a **convolutional neural network**
that detects disease from a leaf photograph, and **A\* search** that plans farm
inspection routes prioritising the plots most at risk. Live weather forecasting
turns those outputs into timed, actionable alerts.

**CS 254 — Introduction to Artificial Intelligence · Team 6 · Ashesi University**

---

## Contents

- [Why this exists](#why-this-exists)
- [Setup](#setup)
- [Running the system](#running-the-system)
- [Usage example](#usage-example)
- [How it works](#how-it-works)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Tests](#tests)
- [Known limitations](#known-limitations)
- [Team](#team)

---

## Why this exists

A smallholder farmer decides daily when to spray, when to irrigate, and which
part of their land to walk. Today that means a weather app, a separate disease
app, and personal experience — with nothing connecting them. Spraying the day
before heavy rain wastes the chemical and the trip; an outbreak on an
uninspected plot spreads before anyone sees it.

FarmerHub connects those signals. A single leaf photograph raises one plot's
risk score, which reroutes the day's inspection walk and moves the farm-wide
health score — all in one interface.

---

## Setup

### 1. Prerequisites

- **Python 3.11**
- **Git LFS** — required. The trained CNN is ~174 MB and is stored via Git Large
  File Storage. Without LFS you will clone a 130-byte pointer file instead of the
  model, and the server will fail on startup.

```bash
# Install Git LFS first (once per machine)
git lfs install
```

macOS: `brew install git-lfs` · Ubuntu: `sudo apt install git-lfs` ·
Windows: bundled with Git for Windows.

### 2. Clone

```bash
git clone https://github.com/Louisa-Lois/G6_FarmerHub.git
cd G6_FarmerHub
git lfs pull        # fetches the CNN model
```

Verify the model downloaded properly — it should be ~174 MB, not ~130 bytes:

On macOS/Linux:

```bash
ls -lh models/farmerhub_disease_model.keras
```

On Windows CMD:

```cmd
dir models\farmerhub_disease_model.keras
```

If Git LFS is working correctly, the model should be approximately 178 MB. A file that is only a few bytes or around 130 bytes indicates that the Git LFS object has not been downloaded correctly.

### 3. Virtual environment

FarmerHub requires Python 3.11.

The virtual environment activation command differs by operating system.

### Windows CMD

Create the virtual environment:

```cmd
py -3.11 -m venv venv
or
python -m venv venv
```

Activate it:

```cmd
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your command prompt.

### Windows PowerShell

Create the virtual environment:

```powershell
py -3.11 -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

### macOS/Linux

Create the virtual environment:

```bash
python3.11 -m venv venv
```

Activate it:

```bash
source venv/bin/activate

## Install Dependencies

After activating the virtual environment, install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Using `python -m pip` ensures that pip runs through the Python interpreter associated with the active virtual environment.

The project uses pinned dependency versions, including:

- FastAPI
- Uvicorn
- scikit-learn
- pandas
- NumPy
- TensorFlow CPU
- Pillow
- python-dotenv
- requests
- pytest

### 4. Weather API key

Weather advice uses the OpenWeatherMap free tier. Get a key at
[openweathermap.org/api](https://openweathermap.org/api), then:

## Configure the Weather API

FarmerHub uses the OpenWeatherMap API for weather-based farming advice.

The API key is not included in the repository. Each user should provide their own API key.

First, create a `.env` file from the provided example.

### Windows CMD

```cmd
copy .env.example .env
```

### macOS/Linux

```bash
cp .env.example .env
```

Open the `.env` file and add your OpenWeatherMap API key:

```env
OWM_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your own OpenWeatherMap API key.

Do not commit your `.env` file or your API key to GitHub.

The `.env.example` file is provided as a template so that other developers know which environment variables they need to configure.

The weather module requires an OpenWeatherMap API key. Other FarmerHub functionality can run without the key. If the weather API is unavailable, the weather functionality returns the application's safer fallback recommendation.

## Running the system

```bash
uvicorn main:app --reload
or 
python -m uvicorn main:app --reload
```

Loading the scikit-learn and TensorFlow models takes roughly 10–15 seconds on
first boot. Then open:

- **http://127.0.0.1:8000** — the FarmerHub dashboard
- **http://127.0.0.1:8000/docs** — interactive API documentation, where every
  endpoint can be exercised without the frontend

---
## Usage example

**1. Register a plot.** In the *My Farm* tab, register a plot at grid
coordinates (0, 0) — Region: Ashanti, District: Amansie West, Crop: Maize.

Only region, district and crop are requested. Soil chemistry and rainfall are
filled in automatically from reference tables, since a smallholder cannot be
expected to know their soil's cation exchange capacity. Any value can still be
overridden with a real soil test.

**2. Scan for disease.** In *Quick Scan*, upload a maize leaf photograph. The CNN
classifies it across 38 categories and converts the result into a risk score for
that plot.

**3. Plan a route.** In *Route Planner*, A\* generates a walking path from the
farm gate that routes *through* high-risk plots rather than around them.

**4. Check farm health.** The dashboard shows a Farm Health Score out of 100,
adjusted by the live weather forecast — for example, adding urgency when a
diseased plot faces rain within 48 hours.

The same prediction via the API:

```bash
curl -X POST http://127.0.0.1:8000/predict-yield \
  -H "Content-Type: application/json" \
  -d '{"region": "ASHANTI", "crop": "MAIZE"}'
```

```json
{
  "predicted_yield_mt_ha": 4.19,
  "range_low_mt_ha": 3.73,
  "range_high_mt_ha": 4.64,
  "national_average_mt_ha": 2.60,
  "confidence": "indicative",
  "basis": "regional averages, not farm-specific measurements"
}
```

---

## How it works

The three models never communicate directly. Each one's output passes through a
connector that normalises it to a **shared risk score between 0.0 and 1.0**, so a
regressor reporting tonnes per hectare and a classifier reporting class
probabilities become comparable.

That shared risk map then drives two consumers:

- **`farm_health_dashboard.py`** blends yield risk and disease risk equally, adds
  a weather modifier (+0.15 for a spray-before-rain window, +0.10 for a dry
  spell), and computes `(1 − average urgency) × 100`.
- **`astar_route.py`** discounts the traversal cost of high-risk plots, so the
  mathematically cheapest path is also the most useful inspection route.

Because A\* uses a non-standard cost function here, the classical admissibility
guarantee could not simply be assumed — it is verified against a Dijkstra
baseline in the test suite.

### API endpoints

15 endpoints, browsable at `/docs`:

| Endpoint | Purpose |
|---|---|
| `POST /predict-yield` | Yield prediction from region and crop |
| `POST /predict-yield-estimate` | Yield estimate with explicit overrides |
| `POST /detect-disease` | Leaf image classification |
| `POST /plan-route` | A\* inspection route |
| `POST /weather-advice` | Forecast-driven alerts |
| `POST /register-plot` | Register a plot to a farm |
| `POST /upload-plot-photo` | Attach a scan to a registered plot |
| `GET /farm-health` | Aggregated Farm Health Score |
| `GET /plots`, `GET /farms` | Registry queries |
| `DELETE /plots/{farm_id}/{row}/{col}` | Remove a plot |
| `GET /districts/{region}`, `GET /crops/{region}/{district}` | Dropdown options |
| `GET /yield-options` | Valid regions and crops |
| `GET /health` | Service health check |

---

## Results

| Module | Metric | Result | Compared against |
|---|---|---|---|
| Yield (Random Forest) | MAE / RMSE / MAPE | 1.101 / 2.139 / 10.86% | Crop-mean lookup: 1.050 / 1.999 / 12.40% |
| Disease (CNN) | Accuracy / weighted F1 | 88% / 0.88 | Chance across 38 classes ≈ 2.6% |
| Route (A\*) | Path cost vs optimal | Exact match (1e-9 tolerance) | Dijkstra, optimal by construction |
| Weather | Scenario tests | 8 / 8 pass | Live OpenWeatherMap forecast |
| System | Automated tests | 22 / 22 pass | — |

**On the yield result.** The Random Forest is marginally *worse* than a crop-mean
lookup on absolute error and better on proportional error, winning on 6 of 11
crops. MAE is dominated by high-tonnage crops — a 4 Mt/Ha miss on cassava (mean
35) outweighs a 0.25 miss on millet (mean 2.4), though the latter is
proportionally worse. With 323 training rows this is the ceiling the data
supports; hyperparameter tuning across 96 configurations moved error by about 1%.

**On evaluation method.** The same district-crop appears in all three years of
the yield data, so a random train/test split leaks near-identical rows across the
divide and reported MAE 0.985. Grouped cross-validation, keeping each
district-crop whole, reported 1.298 — 24% worse, and the honest figure. Every
number above uses the grouped split.

Full methodology, rejected data sources and development decisions are recorded in
[`PROJECT_LOG.md`](PROJECT_LOG.md).

---

## Repository structure

```
main.py                     FastAPI application; 15 endpoints; serves the frontend
core/
  plot_registry.py          SQLite-backed farm database and district defaults
  yield_model.py            Random Forest training and evaluation logic
  yield_service.py          Region/crop lookup wrapper over the trained model
  yield_connector.py        Yield output → 0.0–1.0 risk score
  disease_connector.py      CNN output → 0.0–1.0 risk score
  weather.py                OpenWeatherMap integration and alert generation
  farm_health_dashboard.py  Risk + weather → Farm Health Score
  grid_builder.py           Builds the plot grid from the registry
  astar_route.py            A* search with risk-discounted costs
  route_planner.py          Multi-stop, time-estimated inspection routes
static/index.html           Dashboard UI (vanilla JS, Tailwind CSS)
models/                     Trained .keras and .joblib artefacts, class names
data/                       Cleaned training datasets, plot database
notebooks/                  Training and evaluation notebooks for both models
tests/                      Pytest suite (22 tests)
```

`disease_connector.py` and `yield_connector.py` are deliberately separate so that
code paths needing only a yield prediction never import TensorFlow.

---

## Tests

```bash
pytest tests/ -v
```

22 tests across three suites:

- **Yield service** — verifies the lookup wrapper reproduces the training
  notebook's predictions across all region-crop combinations
- **Registry and dashboard** — plot registration, risk mapping, Farm Health Score
  arithmetic, and isolation between farms
- **A\* router** — path costs checked against an independent Dijkstra
  implementation on a hand-built adversarial grid and ten randomised grids

---

## Known limitations

- **Yield predictions are regional, not farm-specific.** Soil is one 2018 survey
  value per region and rainfall one value per region-year, so two farms in the
  same region receive identical predictions. Interface copy reads *"typical for
  maize in Ashanti"* rather than *"your farm's yield."*
- **Selection bias in the training data.** Ghana's Ministry of Food and
  Agriculture publishes yields only for the ten best-performing districts per
  crop, so the model has never seen an average farm and predicts optimistically —
  4.2 Mt/Ha for maize in Ashanti against a national average of 2.60. Every
  prediction is returned with a range and an `"indicative"` confidence label.
- **The disease model is trained on laboratory images.** PlantVillage photographs
  use uniform backgrounds and controlled lighting. Accuracy on real field photos
  taken with a phone will be lower; field validation is the top priority for
  further work.
- **No authentication.** Any registered plot is reachable by farm ID. Acceptable
  for a prototype, not for real farmers' data.
- **Weather advice is short-horizon.** The OpenWeatherMap free tier provides a
  5-day/3-hour forecast, so all advice is tactical rather than seasonal.

---

## Team

| Member | Module |
|---|---|
| Louisa-Lois Adjoka | Project management, frontend, system integration |
| Kwasi Bekae Ackonor | Computer vision — CNN disease detection |
| Chrishelle Wiafe | Machine learning — yield prediction and evaluation |
| Daniel Ekpale | Search algorithms, backend, testing and deployment |

AI tool use is declared in Appendix A of the final report, per the course's
Responsible Use of AI Tools and Agents policy.
