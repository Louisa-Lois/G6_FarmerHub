"""
FarmerHub AI — Farm Health Dashboard
--------------------------------------
Combines Chrishelle's yield risk, Kwasi's disease risk, and this
project's weather alerts into one farmer-facing view: a per-plot
urgency score + recommendation, and one overall Farm Health Score
for the whole farm (per the proposal: "a single indicator that
allows farmers to quickly assess the overall condition of their
farms").

Per-plot yield+disease fusion reuses Daniel's
grid_builder.build_risk_weights() rather than re-implementing it --
this file only adds the weather layer on top.

Weather is farm-wide, not per-plot (OpenWeatherMap doesn't give
plot-level resolution), so it acts as a uniform modifier: a plot
already flagged as disease-risk gets an urgency boost and a
spray-before-rain recommendation if rain is coming; a plot flagged
as yield-risk gets a boost and an irrigation recommendation if a
dry spell is forecasted.
"""

from core.grid_builder import build_risk_weights


def compute_farm_health(
    yield_risk_map,
    disease_risk_map,
    weather_advice,
    yield_weight=0.5,
    disease_weight=0.5,
    risk_threshold=0.5,
    spray_boost=0.15,
    irrigation_boost=0.10,
    plots_data=None,
):
    """
    yield_risk_map: dict (row,col) -> yield risk 0-1
                     (from yield_connector.get_yield_risk_map)
    disease_risk_map: dict (row,col) -> disease risk 0-1
                     (from disease_connector.get_disease_risk_map;
                     only plots with an uploaded photo appear here)
    weather_advice: dict returned by weather.get_weather_advice()
    plots_data: optional dict (row,col) -> plot metadata dict

    Returns:
      {
        "farm_health_score": 0-100, higher = healthier,
        "plots": { (row,col): {urgency, yield_risk, disease_risk, recommendation, farm_id, crop, region, district, soil_ph} },
        "weather_summary": {...}
      }
    """
    base_risk = build_risk_weights(
        yield_risk_map, disease_risk_map, yield_weight, disease_weight
    )

    rain_soon = weather_advice.get("rain_expected_48h", False)
    irrigation_alert = weather_advice.get("irrigation_alert", False)
    irrigation_msg = weather_advice.get("irrigation_message")

    plots = {}
    for plot, combined_risk in base_risk.items():
        d_risk = disease_risk_map.get(plot, 0.0)
        y_risk = yield_risk_map.get(plot, 0.0)

        urgency = combined_risk
        recommendation = None

        # Spray timing takes priority over irrigation when both would
        # apply -- a closing rain window is more time-sensitive than a
        # multi-day dry spell.
        if d_risk > risk_threshold and rain_soon:
            urgency = min(1.0, urgency + spray_boost)
            recommendation = "Spray before rain — window closing"
        elif y_risk > risk_threshold and irrigation_alert:
            urgency = min(1.0, urgency + irrigation_boost)
            recommendation = irrigation_msg

        plot_meta = plots_data.get(plot, {}) if plots_data else {}

        plots[plot] = {
            "urgency": round(urgency, 3),
            "yield_risk": round(y_risk, 3),
            "disease_risk": round(d_risk, 3),
            "recommendation": recommendation,
            "farm_id": plot_meta.get("farm_id"),
            "crop": plot_meta.get("crop"),
            "region": plot_meta.get("region"),
            "district": plot_meta.get("district"),
            "soil_ph": plot_meta.get("soil_ph"),
        }

    avg_risk = sum(p["urgency"] for p in plots.values()) / len(plots) if plots else 0.0
    farm_health_score = round((1 - avg_risk) * 100, 1)

    return {
        "farm_health_score": farm_health_score,
        "plots": plots,
        "weather_summary": {
            "temp": weather_advice.get("temp", 28.0),
            "humidity": weather_advice.get("humidity", 70),
            "wind_speed": weather_advice.get("wind_speed", 5.0),
            "rain_expected_48h": rain_soon,
            "irrigation_alert": irrigation_alert,
            "planting_recommended": weather_advice.get("planting_recommended", False),
            "planting_message": weather_advice.get("planting_message"),
        },
    }
