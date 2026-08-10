HEAD
# Group-6-Final-Project-FarmerHub
# FarmerHub AI — Backend + Demo Frontend

FastAPI backend for the three core modules, plus a lightweight demo
frontend so you (or Louisa-Lois) can see it working visually before the
real React app exists.

## Project structure

```
farmerhub_backend/
├── main.py                  # FastAPI app -- all endpoints live here
├── requirements.txt
├── core/
│   ├── astar_route.py
│   ├── grid_builder.py
│   ├── route_planner.py
│   ├── yield_connector.py       # yield-only, no TensorFlow needed
│   ├── disease_connector.py     # disease-only, needs TensorFlow
│   ├── integration_connectors.py # combines both, for convenience
│   └── yield_model.py           # Chrishelle's script
├── models/
│   ├── yield_model.joblib
│   ├── disease_model.keras      # currently a CPU-trained prototype
│   └── class_names.json
└── demo-frontend/
    └── index.html            # plain HTML/JS demo page, no build step
```

---

## Part 1 — Backend setup

### First time only

```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### Running it

```bash
uvicorn main:app --reload
```

Wait for `Uvicorn running on http://127.0.0.1:8000` — the first request
after starting can take a few seconds while TensorFlow and the models
finish loading.

Confirm it's alive: open **http://127.0.0.1:8000/docs** for the
interactive Swagger page, or **http://127.0.0.1:8000/health** for a
quick JSON check.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Confirms the server and both models loaded |
| POST | `/predict-yield` | Chrishelle's yield model |
| POST | `/detect-disease` | Disease model (file upload) |
| POST | `/plan-route` | A\* route optimization |
| POST | `/weather-advice` | Rule-based irrigation/planting advice (stretch) |

### Setting up `/weather-advice`

This endpoint calls the OpenWeatherMap API and needs a real API key.

1. Copy `.env.example` to a new file named `.env` in the same folder as `main.py`.
2. Open `.env` and replace `your_key_here` with your real OpenWeatherMap key.
3. **Never commit `.env` or paste a real key into a doc/screenshot/chat** — `.env` is already in `.gitignore`, keep it that way. If a real key has ever been shared anywhere outside your own `.env` file, rotate it on OpenWeatherMap's dashboard.
4. OpenWeatherMap says new keys can take a couple hours to activate — until then this endpoint returns a clean `503` explaining the key isn't set/active yet, not a crash.

Full request/response shapes are in `/docs`, or see the demo frontend's
JavaScript for working examples of every call.

---

## Part 2 — Demo frontend

A single self-contained HTML page (`demo-frontend/index.html`) with a
form for each endpoint. No npm, no React, no build step — just a browser
and a running backend.

### Running it

1. **Keep the backend running** (Part 1, in its own terminal window).
2. In a **second terminal**, serve the demo folder:

```bash
cd demo-frontend
python3 -m http.server 3000
```

3. Open **http://localhost:3000** in your browser.

You should see a green status dot and "Backend is up — 38 disease
classes loaded" at the top. If it's red, the backend isn't running or
isn't reachable — check the first terminal.

### Why port 3000 specifically

The backend's CORS settings (in `main.py`) only allow requests from a
few known origins — `localhost:3000` and `localhost:5173` (React's and
Vite's default dev ports). Serving the demo on `3000` matches that
allowlist, so the browser will actually let the JavaScript talk to the
API. If you serve it on a different port, you'll see CORS errors in the
browser console — either use `3000`/`5173`, or add your port to the
`allow_origins` list in `main.py`.

### What each panel does

- **Predict Yield** — pre-filled with the same example values used in
  testing (Ashanti maize). Edit and submit to see a live prediction.
- **Detect Disease** — upload any leaf photo. Remember the current model
  is a CPU-trained prototype (~52% accuracy on a small subset), so don't
  read too much into wrong predictions yet -- it's there to prove the
  pipeline works end-to-end.
- **Plan Route** — enter a grid size, obstacles, and risk-weighted
  plots, and see the actual computed route and travel time.

### This IS basically what Louisa-Lois will build, just rougher

The demo page's JavaScript (`fetch()` calls with JSON bodies, or
`FormData` for the image upload) is functionally identical to what a
real React app will do to talk to this backend. If you want to preview
how integration will feel before her UI exists, this is it.

---

## Known limitations to fix before final submission

- `disease_model.keras` is a CPU-trained prototype — swap in Kwasi's
  real trained model + `class_names.json` once available. No code
  changes needed, just replace the files in `models/`.
- Per-plot farm data (soil, rainfall, crop, etc.) still needs a real
  source — every test so far has used hand-entered values.
- CORS currently only allows local dev ports — add the real deployed
  frontend URL once one exists.
f5d97d6 (Add backend weather and more)
