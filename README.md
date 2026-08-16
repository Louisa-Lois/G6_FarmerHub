# FarmerHub AI 🌱

**CS 254 — Introduction to Artificial Intelligence (Final Project)**  
**Team Members:** Louisa-Lois Adjoka, Daniel Ekpale, Chrishelle Wiafe, Kwasi Bekae Ackonor

---

## 🚀 Project Overview

FarmerHub AI is an intelligent decision-support system designed to help smallholder farmers in Ghana optimize their daily operations. By consolidating multiple AI techniques into a single, accessible dashboard, the system monitors farm conditions, predicts risks, and generates actionable, weather-aware recommendations.

**Core AI Modules Implemented:**
1. **Yield Prediction (Supervised ML - Random Forest):** Predicts crop yield in tonnes/hectare based on district-level soil chemistry, historical rainfall, and crop type.
2. **Disease Detection (Computer Vision - CNN):** Analyzes uploaded leaf photos to classify diseases across 38 distinct categories and assigns a severity risk score.
3. **Route Optimization (Search - A* Algorithm):** Calculates the most efficient physical inspection route across a farm grid. It utilizes a risk-weighted heuristic to mathematically prioritize detours through diseased or low-yielding plots.

These modules feed into the **Farm Health Dashboard**, which overlays live weather forecasting to deliver urgency alerts (e.g., advising a farmer to delay chemical spraying if heavy rain is imminent).

---

## ⚙️ Setup & Installation

The system is designed for easy reproduction. The Vanilla JS/Tailwind frontend is served directly by the FastAPI backend, meaning **no separate Node.js or npm setup is required**.

### 1. Clone the Repository
```bash
git clone https://github.com/Louisa-Lois/Group-6-Final-Project-FarmerHub.git
cd Group-6-Final-Project-FarmerHub
```

### 2. Create and Activate a Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure the Environment
The Weather Decision Support module requires an OpenWeatherMap API key.

Copy the provided example file:
```bash
cp .env.example .env
```

Open the `.env` file and replace the placeholder with your actual API key:
```env
OWM_API_KEY=your_actual_api_key_here
```

*(Note: If no API key is provided, the weather module includes a robust offline failsafe with fallback data).*

---

## 💻 How to Run the System

With your virtual environment activated, start the FastAPI server:
```bash
uvicorn main:app --reload
```

> **Note:** The server takes approximately 10-15 seconds to boot as it loads the Scikit-Learn Random Forest and TensorFlow CNN models into memory.

Once you see `Application startup complete`, open your web browser and navigate to:
👉 **http://127.0.0.1:8000**

*(Interactive API documentation is also available at **http://127.0.0.1:8000/docs**)*

---

## 🔍 Usage Walkthrough (Grader Guide)

To see all AI modules interacting seamlessly, follow this walkthrough in the web UI:

1. **Register a Farm Plot (Yield Model & Plot Registry):**
   - Click the **"My Farm"** tab.
   - Set coordinates to **Row 0, Column 0**.
   - Select **Region: ASHANTI**, **District: AMANSIE WEST**, **Crop: MAIZE**.
   - Click **Register Plot**. The backend automatically retrieves the soil chemistry for this district and computes baseline yield risk.

2. **Diagnose a Crop (CNN Model):**
   - Click the **"Quick Scan"** tab.
   - Upload an image of a crop leaf (e.g., tomato, corn, potato leaf).
   - Click **Analyze Image**. The CNN returns the predicted disease class, confidence percentage, and computed risk score.

3. **Optimize Inspection Route (A\* Search Algorithm):**
   - Click the **"Route Planner"** tab.
   - Use the **"Obstacle"** mode to place walls or fences on the grid.
   - Click **Plan Optimal Route** (or select a specific Target plot for direct navigation).
   - The algorithm animates the step-by-step optimal path, actively prioritizing detours through high-risk plots.

4. **Review Farm Health (Dashboard Aggregation):**
   - Click the **"Dashboard"** tab.
   - View the unified **Farm Health Score**, priority action alerts, and live weather conditions with actionable agricultural advice.

---

## 🧪 Running the Tests

To verify the logic and mathematical admissibility of the algorithms, run the Pytest suite:
```bash
python -m pytest tests/ -v
```

This suite verifies:
- `test_astar.py`: Adversarial testing confirming the A* heuristic is admissible and matches a ground-truth Dijkstra baseline on complex risk-weighted grids.
- `test_yield_service.py`: Validates the feature engineering pipeline and guarantees prediction consistency with the trained Random Forest model.
- `test_plot_registry_and_farm_health.py`: Verifies multi-farm isolation, district lookup defaults, and end-to-end dashboard health score calculation.

---

## 📂 Repository Structure

* `main.py` — The core FastAPI application, API endpoints, and static file server.
* `requirements.txt` — Pinned environment dependencies required to run the project.
* `README.md` — Master project documentation and setup guide.
* `PROJECT_LOG.md` — Running record of data cleaning decisions, dead ends, and model evaluation history.
* `.env.example` — Template for environment variables (OpenWeatherMap API key).
* `.gitignore` / `.gitattributes` — Git configurations.

**Directories:**
* `core/` — Contains application logic and algorithms:
  * `astar_route.py`, `grid_builder.py`, `route_planner.py` (A* Search & Route Optimization).
  * `yield_model.py`, `yield_service.py`, `yield_connector.py` (Random Forest Yield Prediction).
  * `disease_connector.py` (CNN Disease Detection Adapter).
  * `weather.py` (OpenWeatherMap live decision support).
  * `farm_health_dashboard.py` (Unified Farm Health Score aggregator).
  * `plot_registry.py` (In-memory farm plot database and district defaults).
* `tests/` — Pytest test suite.
* `models/` — Trained `.keras` (CNN) model, `.joblib` (Random Forest) model, and `class_names.json`.
* `data/` — Cleaned training datasets (`farmerhub_yield_training_data_clean.csv`) and model prediction logs.
* `static/` — Frontend user interface (`index.html`).
* `notebooks/` — Jupyter notebooks documenting data cleaning, feature engineering, and model training.