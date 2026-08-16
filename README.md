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
