"""
Region-Based Defaults for Yield Prediction
------------------------------------------------
Most farmers don't know their soil's nitrogen %, phosphorus, CEC, or
even precise seasonal rainfall in mm. This module lets /predict-yield-simple
accept just region + crop + a rough sense of the season, filling in the
rest from regional averages.

IMPORTANT — data provenance:
Only the ASHANTI values below are verified real data (taken directly from
Chrishelle's own worked example in yield_model.py). Every other region's
soil chemistry values are ROUGH ESTIMATES based on Ghana's general
ecological zones (forest belt vs. transition vs. savannah), not sourced
from the actual training data. They exist so the simplified endpoint has
*something* reasonable to fall back on, not because they're verified.

Before this is used for anything beyond a class demo, ask Chrishelle to
export real per-region averages from her training dataset (a simple
`df.groupby('region').mean()` on the same columns) to replace this file.
"""

# Real data point, from yield_model.py's own worked example
_ASHANTI_REAL = dict(
    soil_ph=5.7, organic_matter=6.915, nitrogen=0.26,
    phosphorus=3.49, cec=4.81, region_median_rainfall=1239,
)

# Everything else: rough ecological-zone estimates, NOT verified.
# Forest zone (Ashanti, Western, Western North, Eastern, Central, Bono,
# Bono East, Ahafo): more acidic, higher organic matter, higher rainfall.
# Savannah zone (Northern, Savannah, North East, Upper East, Upper West,
# Oti): less acidic, lower organic matter, lower rainfall.
# Coastal (Greater Accra, Volta): moderate, lower rainfall than forest belt.
REGION_DEFAULTS = {
    "ASHANTI": _ASHANTI_REAL,
    "WESTERN": dict(soil_ph=5.5, organic_matter=7.2, nitrogen=0.28, phosphorus=3.6, cec=5.0, region_median_rainfall=1600),
    "WESTERN NORTH": dict(soil_ph=5.5, organic_matter=7.0, nitrogen=0.27, phosphorus=3.5, cec=4.9, region_median_rainfall=1550),
    "EASTERN": dict(soil_ph=5.8, organic_matter=6.5, nitrogen=0.25, phosphorus=3.3, cec=4.7, region_median_rainfall=1300),
    "CENTRAL": dict(soil_ph=5.9, organic_matter=6.0, nitrogen=0.23, phosphorus=3.1, cec=4.5, region_median_rainfall=1200),
    "BONO": dict(soil_ph=6.0, organic_matter=5.5, nitrogen=0.22, phosphorus=3.0, cec=4.3, region_median_rainfall=1150),
    "BONO EAST": dict(soil_ph=6.0, organic_matter=5.3, nitrogen=0.21, phosphorus=2.9, cec=4.2, region_median_rainfall=1100),
    "AHAFO": dict(soil_ph=5.7, organic_matter=6.3, nitrogen=0.24, phosphorus=3.2, cec=4.6, region_median_rainfall=1350),
    "GREATER ACCRA": dict(soil_ph=6.3, organic_matter=3.5, nitrogen=0.15, phosphorus=2.2, cec=3.5, region_median_rainfall=800),
    "VOLTA": dict(soil_ph=6.2, organic_matter=4.0, nitrogen=0.17, phosphorus=2.4, cec=3.7, region_median_rainfall=1000),
    "OTI": dict(soil_ph=6.4, organic_matter=3.8, nitrogen=0.16, phosphorus=2.3, cec=3.6, region_median_rainfall=1150),
    "NORTHERN": dict(soil_ph=6.5, organic_matter=2.8, nitrogen=0.12, phosphorus=1.8, cec=3.0, region_median_rainfall=1050),
    "SAVANNAH": dict(soil_ph=6.6, organic_matter=2.5, nitrogen=0.11, phosphorus=1.7, cec=2.9, region_median_rainfall=1000),
    "NORTH EAST": dict(soil_ph=6.6, organic_matter=2.6, nitrogen=0.11, phosphorus=1.7, cec=2.9, region_median_rainfall=1000),
    "UPPER EAST": dict(soil_ph=6.7, organic_matter=2.2, nitrogen=0.10, phosphorus=1.5, cec=2.7, region_median_rainfall=950),
    "UPPER WEST": dict(soil_ph=6.7, organic_matter=2.3, nitrogen=0.10, phosphorus=1.6, cec=2.8, region_median_rainfall=1000),
}

# How much higher a well-managed plot's ATTAINABLE yield is versus the
# national average -- "potential_yield" means the ceiling under good
# conditions, not the average, so defaulting it to the national average
# outright would make risk_score come out ~0 for almost every prediction
# (the model routinely predicts above the national mean). This ratio is
# taken from Chrishelle's own verified worked example: potential_yield=5.5
# vs. national_avg[MAIZE]=2.6 -> ~2.115x. Applied to other crops as an
# approximation pending real per-crop attainable-yield data.
POTENTIAL_YIELD_HEADROOM = 2.115

# Which fields are "verified real data" vs "rough estimate", per region --
# surfaced in the API response so nobody mistakes a guess for real data.
VERIFIED_REGIONS = {"ASHANTI"}

# How much to scale the region's median rainfall by, based on the
# farmer's rough sense of how the season has gone -- a stand-in for
# needing an exact mm figure they won't have.
RAINFALL_CONDITION_MULTIPLIERS = {
    "drier": 0.75,
    "normal": 1.0,
    "wetter": 1.25,
}


def get_yield_defaults(region: str, rainfall_condition: str = "normal"):
    """
    Returns (defaults_dict, is_verified, warning) for a given region.

    defaults_dict has all the fields /predict-yield needs except
    region/crop/year/potential_yield: rainfall_mm, soil_ph, organic_matter,
    nitrogen, phosphorus, cec, region_median_rainfall.
    """
    region_key = region.upper()
    base = REGION_DEFAULTS.get(region_key)
    warning = None

    if base is None:
        base = REGION_DEFAULTS["ASHANTI"]
        warning = f"No defaults for region '{region}' -- used Ashanti's as a fallback. Provide exact values for an accurate prediction."

    is_verified = region_key in VERIFIED_REGIONS
    if not is_verified and warning is None:
        warning = (
            f"Soil/rainfall defaults for '{region}' are rough ecological-zone "
            f"estimates, not verified training data. Prediction accuracy is "
            f"lower than for a region with real data (e.g. Ashanti)."
        )

    multiplier = RAINFALL_CONDITION_MULTIPLIERS.get(rainfall_condition, 1.0)
    rainfall_mm = base["region_median_rainfall"] * multiplier

    defaults = dict(base)
    defaults["rainfall_mm"] = rainfall_mm
    return defaults, is_verified, warning
