# FarmerHub AI — Backend

FastAPI backend wiring together the three core modules:

- **Yield Prediction** (Chrishelle's Random Forest) → `POST /predict-yield`
- **Disease Detection** (prototype CNN — swap for Kwasi's final model) → `POST /detect-disease`
- **A\* Route Optimization** (Daniel's module) → `POST /plan-route`

## Project structure

```
farmerhub_backend/
├── main.py                  # FastAPI app -- all endpoints live here
├── requirements.txt
├── core/                    # all module logic
│   ├── astar_route.py
│   ├── grid_builder.py
│   ├── route_planner.py
│   ├── integration_connectors.py
│   └── yield_model.py       # Chrishelle's script
└── models/                  # trained model files
    ├── yield_model.joblib
    ├── disease_model.keras  # currently the CPU-trained prototype
    └── class_names.json
```

## Setup (first time)

```bash
# 1. Create a virtual environment (keeps dependencies isolated)
python3 -m venv venv

# 2. Activate it
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## Running it

```bash
uvicorn main:app --reload
```

You should see something like:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Open **http://127.0.0.1:8000/docs** in a browser — FastAPI auto-generates
an interactive page where you can test every endpoint directly, no
Postman/curl needed.

## Endpoints

### `GET /health`
Quick check that the server and models loaded correctly.

### `POST /predict-yield`
```json
{
  "region": "ASHANTI", "crop": "MAIZE", "year": 2024,
  "rainfall_mm": 1300, "soil_ph": 5.7, "organic_matter": 6.915,
  "nitrogen": 0.26, "phosphorus": 3.49, "cec": 4.81,
  "region_median_rainfall": 1239, "potential_yield": 5.5
}
```
Returns predicted yield (Mt/Ha) and a 0-1 risk score.

### `POST /detect-disease`
Upload a leaf image as form-data (`file` field). Returns predicted
disease class, confidence, and a 0-1 risk score.

### `POST /plan-route`
```json
{
  "rows": 10, "cols": 10,
  "obstacles": [[2,2],[2,3]],
  "start": [0,0],
  "risk_weights": {"4,4": 0.9, "7,8": 0.7, "1,9": 0.3},
  "threshold": 0.6
}
```
Returns the visit order, full walked route, and estimated travel time.

## Known limitations to fix before final submission

- `disease_model.keras` is a CPU-trained prototype (52% val accuracy,
  trained on 60 images/class) — swap in Kwasi's real trained model +
  his `class_names.json` the moment they're available. The endpoint
  code doesn't need to change, just the files in `models/`.
- Per-plot farm data (soil, rainfall, crop, etc. for `/predict-yield`)
  still needs a real source — currently every test has been hand-entered.
- CORS isn't configured yet — needed once Louisa-Lois's frontend
  (React/Flutter) tries to call this from a browser on a different port.
