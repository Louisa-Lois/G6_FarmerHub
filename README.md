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