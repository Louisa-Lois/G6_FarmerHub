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
import os

import joblib
import numpy as np
import requests
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

load_dotenv()

from core.yield_model import predict_yield
from core.yield_service import YieldService
from core.route_planner import yield_to_risk, plan_route, estimate_travel_time
from core.grid_builder import build_grid
from core.weather import get_weather_advice
from core.plot_registry import (
    register_plot, attach_photo, get_plots_data, get_plot_photos,
    get_available_districts, get_available_crops,
)
from core.yield_connector import get_yield_risk_map
from core.disease_connector import get_disease_risk_map
from core.farm_health_dashboard import compute_farm_health

PHOTO_UPLOAD_DIR = "uploaded_plot_photos"
os.makedirs(PHOTO_UPLOAD_DIR, exist_ok=True)

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

# YieldService wraps the same model bundle but looks up soil/rainfall/
# potential-yield reference values itself, so a farmer only has to supply
# region + crop instead of the 8 extra fields /predict-yield needs.
YIELD_SERVICE = YieldService()
DISEASE_MODEL = tf.keras.models.load_model("models/farmerhub_disease_model.keras")

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
# /predict-yield-estimate
# ---------------------------------------------------------------------
# A farmer cannot be expected to know their soil's pH, CEC, or nitrogen
# level, which /predict-yield above requires. This endpoint only needs
# region + crop; YieldService looks up soil, rainfall, and potential-yield
# reference values internally and returns a regional (not farm-specific)
# estimate. See core/yield_service.py for details and caveats.

class YieldEstimateRequest(BaseModel):
    region: str = Field(..., examples=["ASHANTI"])
    crop: str = Field(..., examples=["MAIZE"])
    year: int = Field(default=2024, examples=[2024])
    rainfall_mm: float | None = Field(
        default=None,
        description="Optional forecasted rainfall. Defaults to the region's "
                    "latest observed rainfall if omitted.",
    )


class YieldEstimateResponse(BaseModel):
    predicted_yield_mt_ha: float
    range_low_mt_ha: float
    range_high_mt_ha: float
    national_average_mt_ha: float
    potential_yield_mt_ha: float
    vs_national_average_pct: float
    rainfall_used_mm: float
    rainfall_vs_normal: str
    region: str
    crop: str
    year: int
    risk_score: float
    confidence: str
    basis: str
    caveat: str


@app.post("/predict-yield-estimate", response_model=YieldEstimateResponse)
def predict_yield_estimate_endpoint(req: YieldEstimateRequest):
    try:
        result = YIELD_SERVICE.predict(
            req.region, req.crop, year=req.year, rainfall_mm=req.rainfall_mm
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    risk = yield_to_risk(
        result["predicted_yield_mt_ha"],
        max_expected_yield=result["potential_yield_mt_ha"],
    )
    return YieldEstimateResponse(risk_score=round(risk, 3), **result)


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
# /register-plot, /districts, /crops, /upload-plot-photo
# ---------------------------------------------------------------------
# Lets a farmer register a plot with region + district + crop only.
# plot_registry auto-fills the soil/rainfall fields get_yield_risk_map()
# needs from district-level reference tables, so a farmer never has to
# supply soil pH or CEC by hand. See core/plot_registry.py.

class PlotRegistrationRequest(BaseModel):
    row: int
    col: int
    region: str = Field(..., examples=["ASHANTI"])
    district: str = Field(..., examples=["AMANSIE WEST"])
    crop: str = Field(..., examples=["MAIZE"])
    overrides: dict[str, float] | None = Field(
        default=None,
        description="Optional real soil-test values to override district "
                    "defaults, e.g. {'soil_ph': 6.2}",
    )


class PlotRegistrationResponse(BaseModel):
    row: int
    col: int
    plot_data: dict


@app.post("/register-plot", response_model=PlotRegistrationResponse)
def register_plot_endpoint(req: PlotRegistrationRequest):
    try:
        plot_data = register_plot(
            row=req.row, col=req.col,
            region=req.region, district=req.district, crop=req.crop,
            overrides=req.overrides,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PlotRegistrationResponse(row=req.row, col=req.col, plot_data=plot_data)


@app.get("/districts/{region}")
def list_districts(region: str):
    districts = get_available_districts(region)
    if not districts:
        raise HTTPException(status_code=404, detail=f"No districts found for region '{region}'")
    return {"region": region.upper(), "districts": districts}


@app.get("/crops/{region}/{district}")
def list_crops(region: str, district: str):
    crops = get_available_crops(region, district)
    if not crops:
        raise HTTPException(status_code=404, detail=f"No crops found for '{district}', '{region}'")
    return {"region": region.upper(), "district": district.upper(), "crops": crops}


@app.post("/upload-plot-photo")
async def upload_plot_photo_endpoint(row: int, col: int, file: UploadFile = File(...)):
    contents = await file.read()
    ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    photo_path = os.path.join(PHOTO_UPLOAD_DIR, f"{row}_{col}.{ext}")
    with open(photo_path, "wb") as f:
        f.write(contents)

    try:
        attach_photo(row, col, photo_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"row": row, "col": col, "photo_path": photo_path}


# ---------------------------------------------------------------------
# /farm-health
# ---------------------------------------------------------------------
# Combines every registered plot's yield risk, disease risk (for plots
# with an uploaded photo), and farm-wide weather into one Farm Health
# Score plus a per-plot urgency + recommendation. See
# core/farm_health_dashboard.py.

class PlotHealth(BaseModel):
    urgency: float
    yield_risk: float
    disease_risk: float
    recommendation: str | None


class WeatherSummary(BaseModel):
    rain_expected_48h: bool
    irrigation_alert: bool
    planting_recommended: bool
    planting_message: str | None


class FarmHealthResponse(BaseModel):
    farm_health_score: float
    plots: dict[str, PlotHealth]
    weather_summary: WeatherSummary


@app.get("/farm-health", response_model=FarmHealthResponse)
def farm_health_endpoint(region: str | None = None, town: str | None = None):
    plots_data = get_plots_data()
    if not plots_data:
        raise HTTPException(
            status_code=400,
            detail="No plots registered yet -- register at least one plot first.",
        )

    yield_risk_map = get_yield_risk_map(plots_data, YIELD_MODEL, NATIONAL_AVG)

    plot_photos = get_plot_photos()
    disease_risk_map = (
        get_disease_risk_map(plot_photos, DISEASE_MODEL, CLASS_NAMES, img_size=DISEASE_IMG_SIZE)
        if plot_photos else {}
    )

    # Weather is farm-wide (OpenWeatherMap has no plot-level resolution),
    # so it needs one region. Use the caller's region/town if given,
    # otherwise fall back to whichever region the first registered plot
    # is in -- reasonable for the common case of one farm, one region.
    weather_region = region or next(iter(plots_data.values()))["region"]
    try:
        weather_advice = get_weather_advice(region=weather_region, town=town)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Weather API request failed: {e}")

    result = compute_farm_health(yield_risk_map, disease_risk_map, weather_advice)
    result["plots"] = {f"{r},{c}": v for (r, c), v in result["plots"].items()}

    return FarmHealthResponse(**result)


# ---------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------

@app.get("/yield-options")
def yield_options():
    """Dropdown contents for a region/crop yield-estimate form."""
    return YIELD_SERVICE.options()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "disease_classes_loaded": len(CLASS_NAMES),
        "plots_registered": len(get_plots_data()),
    }
# ---------------------------------------------------------------------
# Frontend Static Serving
# ---------------------------------------------------------------------
# Serves the standalone HTML/JS/CSS frontend to bypass CORS and React
app.mount("/", StaticFiles(directory="static", html=True), name="static")