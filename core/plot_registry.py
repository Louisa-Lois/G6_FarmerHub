"""
FarmerHub AI — Plot Registration
------------------------------------
Lets a farmer register a plot with minimal input (region, district, crop,
grid position) and auto-fills the technical soil/yield fields that
Chrishelle's yield model needs from her own training data -- since a
smallholder farmer won't realistically know their soil's cation exchange
capacity off the top of their head.

Defaults come from farmerhub_yield_training_data_clean.csv, grouped by
district (soil chemistry is constant per district in that dataset) and
by district+crop (potential yield is crop-specific). A farmer with a
real soil test can override any individual field.

NOTE on rainfall: the defaults here are ANNUAL totals (matching the
scale the yield model was trained on), not the same thing as
weather.py's 5-day forecast sums. Don't wire those two together.
"""

import pandas as pd

CURRENT_YEAR = 2026
CSV_PATH = "data/farmerhub_yield_training_data_clean.csv"

_df = pd.read_csv(CSV_PATH)

# District-level soil defaults (constant per district in this dataset)
_SOIL_DEFAULTS = (
    _df.groupby(["region", "district"])[
        [
            "soil_ph_mid",
            "organic_matter_pct_mid",
            "total_nitrogen_pct_mid",
            "avail_phosphorus_mg_kg_mid",
            "cation_exchange_capacity_mid",
        ]
    ]
    .first()
    .to_dict("index")
)

# Potential yield varies by district + crop
_POTENTIAL_YIELD = (
    _df.groupby(["region", "district", "crop"])["potential_yield_mt_ha"]
    .first()
    .to_dict()
)

# Region-level median rainfall (annual scale)
_REGION_RAINFALL = _df.groupby("region")["rainfall_mm"].median().to_dict()

# Which crops actually exist in each district
_CROPS_BY_DISTRICT = (
    _df.groupby(["region", "district"])["crop"].unique().apply(list).to_dict()
)


def get_available_districts(region):
    """Districts that exist for a given region -- for a frontend dropdown."""
    region = region.upper()
    return sorted({d for (r, d) in _SOIL_DEFAULTS if r == region})


def get_available_crops(region, district):
    """Crops actually grown in this district -- for a frontend dropdown,
    so a farmer can't select a crop with no data behind it."""
    key = (region.upper(), district.upper())
    return _CROPS_BY_DISTRICT.get(key, [])


def get_district_defaults(region, district, crop):
    region, district, crop = region.upper(), district.upper(), crop.upper()
    soil_key = (region, district)
    yield_key = (region, district, crop)

    # Safe fallback: If the ML model doesn't know the soil for this district, default to Ashanti
    if soil_key not in _SOIL_DEFAULTS:
        soil_key = ("ASHANTI", "AMANSIE WEST")

    # Safe fallback: If the ML model doesn't know this crop in this district, default to Maize
    if yield_key not in _POTENTIAL_YIELD:
        yield_key = (soil_key[0], soil_key[1], crop)
        if yield_key not in _POTENTIAL_YIELD:
            yield_key = ("ASHANTI", "AMANSIE WEST", "MAIZE")

    soil = _SOIL_DEFAULTS[soil_key]

    return {
        "region": region,
        "district": district,
        "crop": crop,
        "year": CURRENT_YEAR,
        "soil_ph": soil["soil_ph_mid"],
        "organic_matter": soil["organic_matter_pct_mid"],
        "nitrogen": soil["total_nitrogen_pct_mid"],
        "phosphorus": soil["avail_phosphorus_mg_kg_mid"],
        "cec": soil["cation_exchange_capacity_mid"],
        "potential_yield": _POTENTIAL_YIELD[yield_key],
        "region_median_rainfall": _REGION_RAINFALL[region],
        "rainfall_mm": _REGION_RAINFALL[
            region
        ],  # same annual default; farmer can override
    }


# ---------------------------------------------------------------------
# In-memory plot store (simple dict -- fine for a single-farm course demo,
# not meant to survive a backend restart or handle multiple concurrent
# farms; swap for a real database only if that becomes a requirement)
# ---------------------------------------------------------------------

_PLOTS = {}  # (row, col) -> plot dict
_PHOTOS = {}  # (row, col) -> image file path, only for plots with an upload


def register_plot(row, col, region, district, crop, overrides=None, photo_path=None):
    """Registers a plot at (row, col). Auto-fills soil/yield fields from
    district data, then applies any farmer-provided overrides on top."""
    plot_data = get_district_defaults(region, district, crop)
    if overrides:
        plot_data.update(overrides)

    _PLOTS[(row, col)] = plot_data
    if photo_path:
        _PHOTOS[(row, col)] = photo_path

    return plot_data


def attach_photo(row, col, photo_path):
    """Attaches a photo to an already-registered plot. Raises ValueError
    if the plot hasn't been registered yet -- a photo needs a plot to
    belong to."""
    if (row, col) not in _PLOTS:
        raise ValueError(
            f"No plot registered at ({row}, {col}) yet -- register it first."
        )
    _PHOTOS[(row, col)] = photo_path


def get_plots_data():
    """Returns the full plots_data dict in the exact shape
    yield_connector.get_yield_risk_map() expects."""
    return dict(_PLOTS)


def get_plot_photos():
    """Returns the plot_photos dict in the exact shape
    disease_connector.get_disease_risk_map() expects."""
    return dict(_PHOTOS)


def clear_registry():
    """Mainly for testing -- wipes all registered plots."""
    _PLOTS.clear()
    _PHOTOS.clear()
