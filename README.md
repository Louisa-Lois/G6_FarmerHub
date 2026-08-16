# FarmerHub AI 🌱

**CS 254 — Introduction to Artificial Intelligence (Final Project)**
**Team Members:** Louisa-Lois Adjoka, Daniel Ekpale, Chrishelle Wiafe, Kwasi Bekae Ackonor

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
git clone [https://github.com/Louisa-Lois/Group-6-Final-Project-FarmerHub.git](https://github.com/Louisa-Lois/Group-6-Final-Project-FarmerHub.git)
cd Group-6-Final-Project-FarmerHub

### 2. Create and Activate a Virtual Environment

On Mac/Linux:

Bash
python3 -m venv venv
source venv/bin/activate
On Windows:

Bash
python -m venv venv
venv\Scripts\activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Configure the Environment
The Weather Decision Support module requires an OpenWeatherMap API key.

Copy the provided example file:

Bash
cp .env.example .env
Open the .env file and replace the placeholder with your actual API key:

Code snippet
OWM_API_KEY=your_actual_api_key_here
💻 How to Run the System
With your virtual environment activated, start the FastAPI server:

Bash
uvicorn main:app --reload
Note: The server takes approximately 10-15 seconds to boot as it loads the Scikit-Learn Random Forest and TensorFlow CNN models into memory.

Once you see Application startup complete, open your web browser and navigate to:
👉 http://127.0.0.1:8000

(Interactive API documentation is also available at http://127.0.0.1:8000/docs)

🔍 Short Usage Example (Grader Walkthrough)
To see the AI modules interacting, follow this short "happy path" demo in the UI:

Register a Farm Plot (Yield Model):

Click the "My Farm" tab.

Set coordinates to Row 0, Column 0.

Select Region: ASHANTI, District: AMANSIE WEST, Crop: MAIZE.

Click Register Plot. The backend will automatically look up the soil chemistry for this district and generate a baseline yield risk score.

Diagnose a Crop (CNN Model):

Click the "Quick Scan" tab.

Upload an image of a diseased crop leaf (e.g., a Tomato Early Blight leaf).

Click Analyze Image. The CNN will return the disease class, confidence percentage, and an elevated risk score.

Optimize Inspection Route (A Search):*

Click the "Route Planner" tab.

Use the "Obstacle" click mode to place a few gray walls on the grid (simulating sheds or fences).

Click Plan Optimal Route. The algorithm will animate the shortest walkable path that actively detours to visit the high-risk plots identified in Steps 1 and 2.

Review Farm Health (Dashboard Aggregation):

Click the "Dashboard" tab.

View the unified Farm Health Score, which has been mathematically adjusted based on the AI risk scores and the live OpenWeatherMap forecast alerts.

🧪 Running the Tests
To verify the logic and mathematical admissibility of the algorithms, run the included Pytest suite:

Bash
python -m pytest tests/ -v
This suite includes:

test_astar.py: Adversarial testing confirming the A* heuristic is admissible and matches a ground-truth Dijkstra implementation.

test_yield_service.py: Validates the feature engineering pipeline.

test_plot_registry_and_farm_health.py: Ensures end-to-end data formatting between the database, models, and the dashboard aggregator.

## 📂 Repository Structure

* `main.py` — The core FastAPI application, endpoints, and static file server.
* `requirements.txt` — The pinned environment dependencies required to run the project.
* `README.md` — The master project documentation and grader setup guide.
* `PROJECT_LOG.md` — Running record of the team's data cleaning decisions, dead ends, and model evaluation history.
* `.env.example` — Template for the environment variables (e.g., OpenWeatherMap API key).
* `.gitignore` / `.gitattributes` — Git configurations, including LFS tracking for large model files.

**Directories:**
* `core/` — Contains the application logic and algorithms:
  * `astar_route.py`, `grid_builder.py`, `route_planner.py` (A* Search module).
  * `yield_model.py`, `yield_service.py`, `yield_connector.py` (Random Forest module).
  * `disease_connector.py` (CNN module adapter).
  * `weather.py` (OpenWeatherMap API integration).
  * `farm_health_dashboard.py` (Dashboard score aggregator).
  * `plot_registry.py` (In-memory farm database and default mappings).
* `tests/` — The Pytest suite evaluating A* mathematical admissibility, yield data engineering, and end-to-end dashboard integration.
* `models/` — The trained `.keras` (CNN) model, `.joblib` (Random Forest) model, and `class_names.json`.
* `data/` — The cleaned training datasets (`farmerhub_yield_training_data_clean.csv`) and feature importance logs used to establish baselines.
* `static/` — The HTML, CSS, and JS files comprising the frontend interface (`index.html`).
* `notebooks/` — Jupyter notebooks documenting our data cleaning, feature engineering, model training, and weather geocoding tests.