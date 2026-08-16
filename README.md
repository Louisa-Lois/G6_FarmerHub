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
