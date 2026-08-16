"""
FarmerHub AI — Plot Registration & Persistence
------------------------------------------------
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

Plots and uploaded photos are persistently stored in a lightweight
SQLite database (data/farmerhub_plots.db).
"""

import json
import os
import sqlite3
import pandas as pd

CURRENT_YEAR = 2026
CSV_PATH = "data/farmerhub_yield_training_data_clean.csv"
DB_PATH = os.path.join("data", "farmerhub_plots.db")

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

    if soil_key not in _SOIL_DEFAULTS:
        raise ValueError(
            f"No data for district '{district}' in region '{region}'. "
            f"Available districts: {get_available_districts(region)}"
        )
    if yield_key not in _POTENTIAL_YIELD:
        raise ValueError(
            f"'{crop}' is not grown in {district}. "
            f"Available crops here: {get_available_crops(region, district)}"
        )

    soil = _SOIL_DEFAULTS[soil_key]

    return {
        "region": region,
        "district": district,
        "crop": crop,
        "year": CURRENT_YEAR,
        "soil_ph": float(soil["soil_ph_mid"]),
        "organic_matter": float(soil["organic_matter_pct_mid"]),
        "nitrogen": float(soil["total_nitrogen_pct_mid"]),
        "phosphorus": float(soil["avail_phosphorus_mg_kg_mid"]),
        "cec": float(soil["cation_exchange_capacity_mid"]),
        "potential_yield": float(_POTENTIAL_YIELD[yield_key]),
        "region_median_rainfall": float(_REGION_RAINFALL[region]),
        "rainfall_mm": float(_REGION_RAINFALL[region]),  # same annual default; farmer can override
    }


# ---------------------------------------------------------------------
# SQLite Database Persistence Layer
# ---------------------------------------------------------------------

def _get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plots (
                farm_id TEXT NOT NULL,
                row INTEGER NOT NULL,
                col INTEGER NOT NULL,
                region TEXT NOT NULL,
                district TEXT NOT NULL,
                crop TEXT NOT NULL,
                year INTEGER NOT NULL,
                soil_ph REAL NOT NULL,
                organic_matter REAL NOT NULL,
                nitrogen REAL NOT NULL,
                phosphorus REAL NOT NULL,
                cec REAL NOT NULL,
                potential_yield REAL NOT NULL,
                region_median_rainfall REAL NOT NULL,
                rainfall_mm REAL NOT NULL,
                extra_json TEXT,
                PRIMARY KEY (farm_id, row, col)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plot_photos (
                farm_id TEXT NOT NULL,
                row INTEGER NOT NULL,
                col INTEGER NOT NULL,
                photo_path TEXT NOT NULL,
                PRIMARY KEY (farm_id, row, col)
            )
            """
        )
    return conn


def register_plot(
    row, col, region, district, crop, overrides=None, photo_path=None, farm_id="default_farm"
):
    """Registers a plot at (row, col) under farm_id. Auto-fills soil/yield fields
    from district data, then applies any farmer-provided overrides on top."""
    plot_data = get_district_defaults(region, district, crop)
    plot_data["farm_id"] = farm_id
    if overrides:
        plot_data.update(overrides)

    conn = _get_connection()
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO plots (
                farm_id, row, col, region, district, crop, year,
                soil_ph, organic_matter, nitrogen, phosphorus, cec,
                potential_yield, region_median_rainfall, rainfall_mm, extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                farm_id,
                int(row),
                int(col),
                plot_data["region"],
                plot_data["district"],
                plot_data["crop"],
                int(plot_data["year"]),
                float(plot_data["soil_ph"]),
                float(plot_data["organic_matter"]),
                float(plot_data["nitrogen"]),
                float(plot_data["phosphorus"]),
                float(plot_data["cec"]),
                float(plot_data["potential_yield"]),
                float(plot_data["region_median_rainfall"]),
                float(plot_data["rainfall_mm"]),
                json.dumps({k: v for k, v in plot_data.items() if k not in [
                    "farm_id", "row", "col", "region", "district", "crop", "year",
                    "soil_ph", "organic_matter", "nitrogen", "phosphorus", "cec",
                    "potential_yield", "region_median_rainfall", "rainfall_mm"
                ]}),
            ),
        )
        if photo_path:
            conn.execute(
                """
                INSERT OR REPLACE INTO plot_photos (farm_id, row, col, photo_path)
                VALUES (?, ?, ?, ?)
                """,
                (farm_id, int(row), int(col), photo_path),
            )
    conn.close()
    return plot_data


def attach_photo(row, col, photo_path, farm_id="default_farm"):
    """Attaches a photo to an already-registered plot under farm_id. Raises ValueError
    if the plot hasn't been registered yet -- a photo needs a plot to belong to."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM plots WHERE farm_id = ? AND row = ? AND col = ?",
        (farm_id, int(row), int(col)),
    )
    if not cur.fetchone():
        conn.close()
        raise ValueError(
            f"No plot registered at ({row}, {col}) for farm '{farm_id}' yet -- register it first."
        )

    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO plot_photos (farm_id, row, col, photo_path)
            VALUES (?, ?, ?, ?)
            """,
            (farm_id, int(row), int(col), photo_path),
        )
    conn.close()


def get_plots_data(farm_id="default_farm"):
    """Returns the plots_data dict {(row, col): plot_data} for the specified farm_id
    in the exact shape yield_connector.get_yield_risk_map() expects.
    If farm_id is None or 'all', returns all registered plots."""
    conn = _get_connection()
    cur = conn.cursor()
    if farm_id is None or farm_id == "all":
        cur.execute("SELECT * FROM plots")
    else:
        cur.execute("SELECT * FROM plots WHERE farm_id = ?", (farm_id,))
    rows = cur.fetchall()
    conn.close()

    result = {}
    for r in rows:
        plot_dict = {
            "farm_id": r["farm_id"],
            "region": r["region"],
            "district": r["district"],
            "crop": r["crop"],
            "year": r["year"],
            "soil_ph": r["soil_ph"],
            "organic_matter": r["organic_matter"],
            "nitrogen": r["nitrogen"],
            "phosphorus": r["phosphorus"],
            "cec": r["cec"],
            "potential_yield": r["potential_yield"],
            "region_median_rainfall": r["region_median_rainfall"],
            "rainfall_mm": r["rainfall_mm"],
        }
        if r["extra_json"]:
            try:
                extra = json.loads(r["extra_json"])
                plot_dict.update(extra)
            except Exception:
                pass
        result[(r["row"], r["col"])] = plot_dict
    return result


def get_plot_photos(farm_id="default_farm"):
    """Returns the plot_photos dict {(row, col): photo_path} for the specified farm_id
    in the exact shape disease_connector.get_disease_risk_map() expects.
    If farm_id is None or 'all', returns all plot photos."""
    conn = _get_connection()
    cur = conn.cursor()
    if farm_id is None or farm_id == "all":
        cur.execute("SELECT row, col, photo_path FROM plot_photos")
    else:
        cur.execute(
            "SELECT row, col, photo_path FROM plot_photos WHERE farm_id = ?",
            (farm_id,),
        )
    rows = cur.fetchall()
    conn.close()
    return {(r["row"], r["col"]): r["photo_path"] for r in rows}


def get_available_farms():
    """Returns a sorted list of registered farm IDs."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT farm_id FROM plots ORDER BY farm_id")
    rows = cur.fetchall()
    conn.close()
    farms = [r["farm_id"] for r in rows]
    return farms if farms else ["default_farm"]


def list_plots(farm_id=None):
    """Returns a list of all plots with photo metadata.
    If farm_id is None or 'all', returns all registered plots across all farms."""
    conn = _get_connection()
    cur = conn.cursor()
    if farm_id is None or farm_id == "all":
        cur.execute(
            """
            SELECT p.*, pp.photo_path
            FROM plots p
            LEFT JOIN plot_photos pp ON p.farm_id = pp.farm_id AND p.row = pp.row AND p.col = pp.col
            ORDER BY p.farm_id, p.row, p.col
            """
        )
    else:
        cur.execute(
            """
            SELECT p.*, pp.photo_path
            FROM plots p
            LEFT JOIN plot_photos pp ON p.farm_id = pp.farm_id AND p.row = pp.row AND p.col = pp.col
            WHERE p.farm_id = ?
            ORDER BY p.row, p.col
            """,
            (farm_id,),
        )
    rows = cur.fetchall()
    conn.close()

    result = []
    for r in rows:
        item = {
            "farm_id": r["farm_id"],
            "row": r["row"],
            "col": r["col"],
            "region": r["region"],
            "district": r["district"],
            "crop": r["crop"],
            "year": r["year"],
            "soil_ph": r["soil_ph"],
            "organic_matter": r["organic_matter"],
            "nitrogen": r["nitrogen"],
            "phosphorus": r["phosphorus"],
            "cec": r["cec"],
            "potential_yield": r["potential_yield"],
            "region_median_rainfall": r["region_median_rainfall"],
            "rainfall_mm": r["rainfall_mm"],
            "photo_path": r["photo_path"],
        }
        if r["extra_json"]:
            try:
                extra = json.loads(r["extra_json"])
                item.update(extra)
            except Exception:
                pass
        result.append(item)
    return result


def delete_plot(row, col, farm_id="default_farm"):
    """Deletes a plot and any attached photo at (row, col) under farm_id."""
    conn = _get_connection()
    with conn:
        conn.execute(
            "DELETE FROM plots WHERE farm_id = ? AND row = ? AND col = ?",
            (farm_id, int(row), int(col)),
        )
        conn.execute(
            "DELETE FROM plot_photos WHERE farm_id = ? AND row = ? AND col = ?",
            (farm_id, int(row), int(col)),
        )
    conn.close()


def clear_registry(farm_id=None):
    """Mainly for testing -- wipes registered plots from the database."""
    conn = _get_connection()
    with conn:
        if farm_id is None:
            conn.execute("DELETE FROM plots")
            conn.execute("DELETE FROM plot_photos")
        else:
            conn.execute("DELETE FROM plots WHERE farm_id = ?", (farm_id,))
            conn.execute("DELETE FROM plot_photos WHERE farm_id = ?", (farm_id,))
    conn.close()
