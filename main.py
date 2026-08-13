"""
FarmerHub AI — Backend
--------------------------
FastAPI app exposing the three core modules as HTTP endpoints:
  POST /predict-yield   -> Chrishelle's yield model
  POST /detect-disease  -> Kwasi's disease model (prototype for now)
  POST /plan-route      -> Daniel's A* route optimization

Run with:  uvicorn main:app --reload
Docs at:   http://127.0.0.1:8000/docs
"""

import io
import json

import joblib
import numpy as np
import requests
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

load_dotenv()

from core.yield_model import predict_yield
from core.route_planner import yield_to_risk, plan_route, estimate_travel_time
from core.grid_builder import build_grid
from core.weather import get_weather_advice

app = FastAPI(title="FarmerHub AI Backend", version="0.1.0")

# Allows a React dev server (localhost:3000 is Create React App's default,
# 5173 is Vite's) to call this API from the browser. Add the real deployed
# frontend URL here too once Louisa-Lois has one.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Load all models ONCE at startup (not per-request -- that would be slow)
# ---------------------------------------------------------------------

_yield_bundle = joblib.load("models/yield_model.joblib")
YIELD_MODEL = _yield_bundle["model"]
NATIONAL_AVG = _yield_bundle["national_avg"]

DISEASE_MODEL = tf.keras.models.load_model("models/disease_model.keras")
with open("models/class_names.json") as f:
    CLASS_NAMES = json.load(f)

DISEASE_IMG_SIZE = (256, 256)  # must match what the model was trained on -- Kwasi's real model, verified via model.input_shape


# ---------------------------------------------------------------------
# /predict-yield
# ---------------------------------------------------------------------

class YieldRequest(BaseModel):
    region: str = Field(..., examples=["ASHANTI"])
    crop: str = Field(..., examples=["MAIZE"])
    year: int = Field(..., examples=[2024])
    rainfall_mm: float
    soil_ph: float
    organic_matter: float
    nitrogen: float
    phosphorus: float
    cec: float
    region_median_rainfall: float
    potential_yield: float

    @field_validator("region_median_rainfall", "potential_yield")
    @classmethod
    def must_be_positive(cls, v, info):
        if v <= 0:
            raise ValueError(f"{info.field_name} must be greater than 0 (got {v})")
        return v


class YieldResponse(BaseModel):
    predicted_yield_mt_ha: float
    risk_score: float


@app.post("/predict-yield", response_model=YieldResponse)
def predict_yield_endpoint(req: YieldRequest):
    try:
        predicted = predict_yield(
            YIELD_MODEL, NATIONAL_AVG,
            region=req.region, crop=req.crop, year=req.year,
            rainfall_mm=req.rainfall_mm, soil_ph=req.soil_ph,
            organic_matter=req.organic_matter, nitrogen=req.nitrogen,
            phosphorus=req.phosphorus, cec=req.cec,
            region_median_rainfall=req.region_median_rainfall,
            potential_yield=req.potential_yield,
        )
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown crop '{req.crop}' -- not in the trained model's national average table")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Yield prediction failed: {e}")

    risk = yield_to_risk(predicted, max_expected_yield=req.potential_yield)
    return YieldResponse(predicted_yield_mt_ha=round(predicted, 3), risk_score=round(risk, 3))


# ---------------------------------------------------------------------
# /detect-disease
# ---------------------------------------------------------------------

class DiseaseResponse(BaseModel):
    predicted_class: str
    confidence: float
    risk_score: float


@app.post("/detect-disease", response_model=DiseaseResponse)
async def detect_disease_endpoint(file: UploadFile = File(...)):
    contents = await file.read()

    # Decode directly from bytes rather than round-tripping through a temp
    # file: NamedTemporaryFile keeps an exclusive lock on Windows, so
    # re-opening that same path (as tf.keras.utils.load_img would) fails
    # there with a PermissionError, even though it works fine on Linux/Mac.
    try:
        img = tf.keras.utils.load_img(io.BytesIO(contents), target_size=DISEASE_IMG_SIZE)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as an image")

    arr = tf.keras.utils.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = DISEASE_MODEL.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    top_class = CLASS_NAMES[top_idx]
    confidence = float(preds[top_idx])
    risk = 0.0 if "healthy" in top_class.lower() else confidence

    return DiseaseResponse(
        predicted_class=top_class,
        confidence=round(confidence, 3),
        risk_score=round(risk, 3),
    )


# ---------------------------------------------------------------------
# /plan-route
# ---------------------------------------------------------------------

class RouteRequest(BaseModel):
    rows: int
    cols: int
    obstacles: list[list[int]] = Field(default_factory=list, description="[[row, col], ...]")
    start: list[int] = Field(..., description="[row, col]")
    risk_weights: dict[str, float] = Field(
        ..., description='Keys as "row,col" strings, e.g. {"3,4": 0.82}')
    threshold: float = 0.6
    plot_size_meters: float = 8.0
    walking_speed_m_per_min: float = 60.0

    @field_validator("rows", "cols")
    @classmethod
    def dims_must_be_positive(cls, v, info):
        if v <= 0:
            raise ValueError(f"{info.field_name} must be greater than 0 (got {v})")
        return v


class RouteResponse(BaseModel):
    stops: list[list[int]]
    route: list[list[int]]
    estimated_minutes: float


@app.post("/plan-route", response_model=RouteResponse)
def plan_route_endpoint(req: RouteRequest):
    if len(req.start) != 2:
        raise HTTPException(status_code=400, detail=f"'start' must be [row, col], got {req.start}")

    start = tuple(req.start)
    if not (0 <= start[0] < req.rows and 0 <= start[1] < req.cols):
        raise HTTPException(
            status_code=400,
            detail=f"'start' {list(start)} is outside the {req.rows}x{req.cols} grid",
        )

    obstacles = [tuple(o) for o in req.obstacles]
    grid = build_grid(req.rows, req.cols, obstacles=obstacles)
    if not grid.get(start, False):
        raise HTTPException(status_code=400, detail=f"'start' {list(start)} is on an obstacle")

    risk_weights = {}
    for key, val in req.risk_weights.items():
        parts = key.split(",")
        if len(parts) != 2:
            raise HTTPException(
                status_code=400,
                detail=f'risk_weights key "{key}" is not in "row,col" format (e.g. "3,4")',
            )
        try:
            r, c = int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f'risk_weights key "{key}" has non-integer row/col',
            )
        risk_weights[(r, c)] = val

    priority_plots = [p for p, r in risk_weights.items()
                       if r >= req.threshold and grid.get(p, False)]

    result = plan_route(grid, start, priority_plots, risk_weights)
    minutes = estimate_travel_time(result["route"], req.plot_size_meters, req.walking_speed_m_per_min)

    return RouteResponse(
        stops=[list(p) for p in result["stops"]],
        route=[list(p) for p in result["route"]],
        estimated_minutes=round(minutes, 1),
    )


# ---------------------------------------------------------------------
# /weather-advice
# ---------------------------------------------------------------------

class WeatherRequest(BaseModel):
    region: str = Field(..., examples=["ASHANTI"])
    town: str | None = Field(default=None, examples=["Kumasi,GH"])


class WeatherResponse(BaseModel):
    location_name: str
    rain_expected_48h: bool
    rain_amount_mm: float
    dry_days_forecasted: int
    irrigation_alert: bool
    irrigation_message: str | None
    planting_recommended: bool
    planting_message: str | None
    location_warning: str | None


@app.post("/weather-advice", response_model=WeatherResponse)
def weather_advice_endpoint(req: WeatherRequest):
    try:
        advice = get_weather_advice(region=req.region, town=req.town)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Weather API request failed: {e}")
    return WeatherResponse(**advice)


# ---------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "disease_classes_loaded": len(CLASS_NAMES)}
