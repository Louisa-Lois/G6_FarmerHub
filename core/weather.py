"""
FarmerHub AI — Weather-Based Decision Support (stretch module)
--------------------------------------------------------------------
Uses OpenWeatherMap's 5-day/3-hour forecast endpoint (NOT the current-
conditions endpoint -- that only gives right-now weather, and irrigation
/ planting advice needs to look 48h ahead) to produce rule-based advice.
No ML here, per the proposal -- this is deliberately simple logic.
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OWM_API_KEY")

FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

# Farmers register by region, not GPS coordinates, so we map each region
# to a representative town for the API call. Rough approximation --
# regional weather varies internally, which is why get_weather_advice
# also accepts an explicit `town` override.
REGION_TOWNS = {
    "ASHANTI": "Kumasi,GH",
    "WESTERN": "Sekondi,GH",
    "WESTERN NORTH": "Sefwi Wiawso,GH",
    "EASTERN": "Koforidua,GH",
    "CENTRAL": "Cape Coast,GH",
    "GREATER ACCRA": "Accra,GH",
    "VOLTA": "Ho,GH",
    "OTI": "Dambai,GH",
    "NORTHERN": "Tamale,GH",
    "SAVANNAH": "Damongo,GH",
    "NORTH EAST": "Nalerigu,GH",
    "UPPER EAST": "Bolgatanga,GH",
    "UPPER WEST": "Wa,GH",
    "BONO": "Sunyani,GH",
    "BONO EAST": "Techiman,GH",
    "AHAFO": "Goaso,GH",
}

RAIN_THRESHOLD_MM = 1.0   # below this, treated as "no meaningful rain"
FORECAST_SLOTS_48H = 16   # OpenWeatherMap gives 3-hour steps -> 16 = 48h


def get_weather_advice(region: str, town: str | None = None) -> dict:
    """
    Returns rule-based irrigation and planting advice for a region/town,
    based on the next 48 hours of forecast data.
    """
    if not API_KEY:
        raise RuntimeError(
            "OWM_API_KEY is not set. Create a .env file next to main.py "
            "with OWM_API_KEY=your_key_here (see README)."
        )

    location_warning = None
    query = town if town else REGION_TOWNS.get(region.upper())
    if not query:
        query = REGION_TOWNS["GREATER ACCRA"]
        location_warning = (
            f"No known town mapping for region '{region}' -- "
            f"defaulted to Accra. Pass an explicit 'town' for accuracy."
        )

    resp = requests.get(
        FORECAST_URL,
        params={"q": query, "appid": API_KEY, "units": "metric"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    slots = data.get("list", [])[:FORECAST_SLOTS_48H]
    if not slots:
        raise RuntimeError(f"No forecast data returned for '{query}'")

    rain_amount_mm = sum(slot.get("rain", {}).get("3h", 0.0) for slot in slots)
    rain_expected_48h = rain_amount_mm >= RAIN_THRESHOLD_MM

    # Count consecutive dry days from now, using calendar date per slot
    dry_days_forecasted = 0
    seen_dates = {}
    for slot in slots:
        date = slot["dt_txt"].split(" ")[0]
        rained = slot.get("rain", {}).get("3h", 0.0) >= RAIN_THRESHOLD_MM
        seen_dates.setdefault(date, False)
        if rained:
            seen_dates[date] = True
    for date in sorted(seen_dates):
        if seen_dates[date]:
            break
        dry_days_forecasted += 1

    irrigation_alert = not rain_expected_48h
    irrigation_message = (
        "No significant rain expected in the next 48 hours -- irrigation recommended."
        if irrigation_alert else None
    )

    # Simple rule: recommend planting if rain is coming but not so much
    # it risks flooding/waterlogging young plants
    planting_recommended = rain_expected_48h and rain_amount_mm < 40.0
    if rain_expected_48h and rain_amount_mm >= 40.0:
        planting_message = f"Heavy rain expected (~{rain_amount_mm:.0f}mm) -- consider delaying planting to avoid waterlogging."
    elif planting_recommended:
        planting_message = f"Moderate rain expected (~{rain_amount_mm:.0f}mm) in the next 48 hours -- good conditions for planting."
    else:
        planting_message = None

    return dict(
        location_name=query,
        rain_expected_48h=rain_expected_48h,
        rain_amount_mm=round(rain_amount_mm, 1),
        dry_days_forecasted=dry_days_forecasted,
        irrigation_alert=irrigation_alert,
        irrigation_message=irrigation_message,
        planting_recommended=planting_recommended,
        planting_message=planting_message,
        location_warning=location_warning,
    )
