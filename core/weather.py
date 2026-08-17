"""
FarmerHub AI — Weather-Based Decision Support
-----------------------------------------------
Turns live OpenWeatherMap forecast data into farmer-facing alerts:
  - Rain expected in the next 48h (for spray-timing decisions)
  - Dry-spell detection -> irrigation alerts
  - Favorable-rain detection -> planting window recommendations

Location can be given as a Ghana region name (mapped to its capital
via REGION_TO_TOWN) or a specific town/city name.

Note: the free OpenWeatherMap tier only provides a 5-day/3-hour
forecast, not historical data -- so this gives short-term tactical
signals ("irrigate this week"), not seasonal planting guidance
("the rainy season is starting").
"""

import os
import math
import requests
from collections import defaultdict

API_KEY = os.getenv("OWM_API_KEY")

REGION_TO_TOWN = {
    "AHAFO": "Goaso",
    "ASHANTI": "Kumasi",
    "BONO": "Sunyani",
    "BONO EAST": "Techiman",
    "CENTRAL": "Cape Coast",
    "EASTERN": "Koforidua",
    "GREATER ACCRA": "Accra",
    "NORTHEAST": "Nalerigu",
    "NORTHERN": "Tamale",
    "OTI": "Dambai",
    "SAVANNAH": "Damongo",
    "UPPER EAST": "Bolgatanga",
    "UPPER WEST": "Wa",
    "VOLTA": "Ho",
    "WESTERN": "Sekondi",
    "WESTERN NORTH": "Sefwi Wiawso",
}


# ---------------------------------------------------------------------
# Geocoding + forecast fetch
# ---------------------------------------------------------------------


def geocode_location(name, api_key=None, country_code="GH"):
    """Converts a place name into lat/lon using OpenWeatherMap's Geocoding API."""
    api_key = api_key or API_KEY
    resp = requests.get(
        "https://api.openweathermap.org/geo/1.0/direct",
        params={"q": f"{name},{country_code}", "limit": 1, "appid": api_key},
    )
    results = resp.json()
    if not results:
        return None, None
    return results[0]["lat"], results[0]["lon"]


def get_forecast(lat, lon, api_key=None):
    api_key = api_key or API_KEY
    resp = requests.get(
        "https://api.openweathermap.org/data/2.5/forecast",
        params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
    )
    return resp.json()


def resolve_location(location_name, api_key=None):
    """Accepts either a Ghana region name (mapped to its capital via
    REGION_TO_TOWN) or a specific town/city name directly."""
    town = REGION_TO_TOWN.get(location_name.upper())
    query = town if town else location_name
    return geocode_location(query, api_key)


# ---------------------------------------------------------------------
# Location cross-check (region dropdown + optional town field)
# ---------------------------------------------------------------------


def haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line distance between two coordinates, in km."""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def validate_town_matches_region(
    town_name, region_name, api_key=None, threshold_km=100
):
    api_key = api_key or API_KEY
    town_lat, town_lon = geocode_location(town_name, api_key)
    if town_lat is None:
        return False, None, f"Could not find '{town_name}' — check the spelling."

    region_capital = REGION_TO_TOWN.get(region_name.upper())
    if region_capital is None:
        return False, None, f"'{region_name}' is not a recognized region."

    region_lat, region_lon = geocode_location(region_capital, api_key)
    distance = haversine_km(town_lat, town_lon, region_lat, region_lon)

    if distance > threshold_km:
        return (
            False,
            distance,
            (
                f"'{town_name}' looks far from {region_name} region "
                f"(~{distance:.0f}km from {region_capital}). Please double-check."
            ),
        )
    return True, distance, None


# ---------------------------------------------------------------------
# Core weather signals
# ---------------------------------------------------------------------


def _daily_rain_totals(forecast_json):
    """Groups the 5-day/3-hour forecast into full-day totals. Only
    returns days with all 8 blocks present -- partial days at the start
    and end of the forecast window don't have enough data to judge fairly."""
    daily_rain = defaultdict(float)
    daily_block_count = defaultdict(int)
    for block in forecast_json["list"]:
        date = block["dt_txt"].split(" ")[0]
        rain_mm = block.get("rain", {}).get("3h", 0)
        daily_rain[date] += rain_mm
        daily_block_count[date] += 1
    return {d: total for d, total in daily_rain.items() if daily_block_count[d] == 8}


def check_rain_expected(forecast_json, hours=48, pop_threshold=0.5):
    """Returns True if any upcoming 3h block has a high chance of rain."""
    blocks_needed = hours // 3
    upcoming = forecast_json["list"][:blocks_needed]
    for block in upcoming:
        if block["pop"] >= pop_threshold:
            rain_mm = block.get("rain", {}).get("3h", 0)
            return True, block["dt_txt"], rain_mm
    return False, None, 0


def check_dry_spell(forecast_json, daily_threshold_mm=1.0):
    """Counts how many full days are forecasted as dry."""
    full_days = _daily_rain_totals(forecast_json)
    dry_days = [d for d, total in full_days.items() if total < daily_threshold_mm]
    return len(dry_days), full_days


# ---------------------------------------------------------------------
# Farmer-facing alerts
# ---------------------------------------------------------------------


def get_irrigation_alert(forecast_json, dry_day_threshold=3):
    dry_count, _ = check_dry_spell(forecast_json)
    if dry_count >= dry_day_threshold:
        return (
            True,
            f"Irrigate now — {dry_count} dry day(s) forecasted with no significant rain.",
        )
    return False, None


def check_planting_window(
    forecast_json, min_moderate_days=2, heavy_mm=15.0, light_mm=1.0
):
    full_days = _daily_rain_totals(forecast_json)
    moderate_days = [d for d, t in full_days.items() if light_mm <= t <= heavy_mm]
    heavy_days = [d for d, t in full_days.items() if t > heavy_mm]

    if heavy_days:
        return False, "Heavy rain expected — risk of seed washout, hold off planting."
    elif len(moderate_days) >= min_moderate_days:
        return (
            True,
            f"Good window to plant — steady rain expected over the next {len(moderate_days)} day(s).",
        )
    else:
        return False, "Too dry for planting right now — irrigate first."

    # ---------------------------------------------------------------------
    # Single entry point for the API endpoint
    # ---------------------------------------------------------------------


def get_weather_advice(region, town=None, api_key=None):
    """Bundles location resolution + all four signals into one response
    dict, ready to be wrapped by the FastAPI endpoint in main.py."""
    api_key = api_key or API_KEY
    location_warning = None

    try:
        if town:
            is_consistent, _, warning = validate_town_matches_region(
                town, region, api_key
            )
            location_warning = warning if not is_consistent else None
            lat, lon = geocode_location(town, api_key)
        else:
            lat, lon = resolve_location(region, api_key)

        forecast_data = get_forecast(lat, lon, api_key)

        # Extract current conditions from the first 3-hour forecast block
        current = forecast_data.get("list", [{}])[0]
        temp = current.get("main", {}).get("temp", 28.0)
        humidity = current.get("main", {}).get("humidity", 70)
        wind_speed = current.get("wind", {}).get("speed", 5.0)

        rain_soon, _, rain_amount = check_rain_expected(forecast_data)
        dry_count, _ = check_dry_spell(forecast_data)
        should_irrigate, irrigation_msg = get_irrigation_alert(forecast_data)
        can_plant, planting_msg = check_planting_window(forecast_data)
        location_name = forecast_data.get("city", {}).get("name", town or region)

    except Exception as e:
        print(f"WEATHER API ERROR: {e}")
        # Failsafe if API key is missing, invalid, or network is blocked
        location_name = town or region
        rain_soon = False
        rain_amount = 0.0
        dry_count = 5
        should_irrigate = True
        irrigation_msg = "Irrigate now — Dry spell detected (Fallback data)."
        can_plant = False
        planting_msg = "Too dry for planting right now (Fallback data)."
        temp = 28.0
        humidity = 70
        wind_speed = 5.0

    return {
        "location_name": location_name,
        "temp": round(temp, 1),
        "humidity": humidity,
        "wind_speed": round(wind_speed, 1),
        "rain_expected_48h": rain_soon,
        "rain_amount_mm": round(rain_amount, 2),
        "dry_days_forecasted": dry_count,
        "irrigation_alert": should_irrigate,
        "irrigation_message": irrigation_msg,
        "planting_recommended": can_plant,
        "planting_message": planting_msg,
        "location_warning": location_warning,
    }
