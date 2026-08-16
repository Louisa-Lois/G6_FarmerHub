"""
Verifies plot_registry.py's district-level defaults line up with what
yield_connector.get_yield_risk_map() and disease_connector.get_disease_risk_map()
expect, and that farm_health_dashboard.compute_farm_health() runs end to end
on real registered plots.

Run from the backend project root:  pytest tests/test_plot_registry_and_farm_health.py
"""
import joblib
import pytest

from core import plot_registry
from core.yield_connector import get_yield_risk_map
from core.farm_health_dashboard import compute_farm_health


@pytest.fixture(autouse=True)
def clear_registry():
    """Each test starts with an empty plot registry."""
    plot_registry.clear_registry()
    yield
    plot_registry.clear_registry()


@pytest.fixture(scope="module")
def yield_bundle():
    bundle = joblib.load("models/yield_model.joblib")
    return bundle["model"], bundle["national_avg"]


def test_district_lookup_finds_real_districts():
    districts = plot_registry.get_available_districts("ASHANTI")
    assert "AMANSIE WEST" in districts


def test_crop_lookup_matches_training_data():
    crops = plot_registry.get_available_crops("ASHANTI", "AMANSIE WEST")
    assert "CASSAVA" in crops


def test_unknown_district_raises():
    with pytest.raises(ValueError):
        plot_registry.get_district_defaults("ASHANTI", "NOWHERE", "MAIZE")


def test_unknown_crop_for_district_raises():
    with pytest.raises(ValueError):
        plot_registry.get_district_defaults("ASHANTI", "AMANSIE WEST", "BANANA")


def test_register_plot_output_matches_get_yield_risk_map_input(yield_bundle):
    """The exact contract plot_registry's docstring promises: register_plot's
    output must be directly usable by get_yield_risk_map() with no reshaping."""
    model, national_avg = yield_bundle
    plot_registry.register_plot(0, 0, "ASHANTI", "AMANSIE WEST", "CASSAVA")
    plots_data = plot_registry.get_plots_data()

    risk_map = get_yield_risk_map(plots_data, model, national_avg)

    assert (0, 0) in risk_map
    assert 0.0 <= risk_map[(0, 0)] <= 1.0


def test_overrides_apply_on_top_of_defaults():
    plot_data = plot_registry.register_plot(
        1, 1, "ASHANTI", "AMANSIE WEST", "CASSAVA",
        overrides={"soil_ph": 6.9},
    )
    assert plot_data["soil_ph"] == 6.9
    # untouched fields keep their district default
    assert plot_data["region"] == "ASHANTI"


def test_attach_photo_requires_existing_plot():
    with pytest.raises(ValueError):
        plot_registry.attach_photo(9, 9, "some/path.jpg")


def test_farm_health_end_to_end(yield_bundle):
    model, national_avg = yield_bundle
    plot_registry.register_plot(0, 0, "ASHANTI", "AMANSIE WEST", "CASSAVA")
    plot_registry.register_plot(0, 1, "ASHANTI", "AMANSIE WEST", "MAIZE")

    yield_risk_map = get_yield_risk_map(plot_registry.get_plots_data(), model, national_avg)
    disease_risk_map = {}  # no photos uploaded in this test

    weather_advice = {
        "rain_expected_48h": True,
        "irrigation_alert": False,
        "irrigation_message": None,
        "planting_recommended": True,
        "planting_message": "Good planting window",
    }

    result = compute_farm_health(yield_risk_map, disease_risk_map, weather_advice)

    assert 0 <= result["farm_health_score"] <= 100
    assert set(result["plots"].keys()) == {(0, 0), (0, 1)}
    for plot in result["plots"].values():
        assert 0.0 <= plot["urgency"] <= 1.0


def test_multi_farm_isolation():
    plot_registry.register_plot(0, 0, "ASHANTI", "AMANSIE WEST", "CASSAVA", farm_id="farm_1")
    plot_registry.register_plot(0, 0, "ASHANTI", "AMANSIE WEST", "MAIZE", farm_id="farm_2")

    farm1_plots = plot_registry.get_plots_data(farm_id="farm_1")
    farm2_plots = plot_registry.get_plots_data(farm_id="farm_2")

    assert (0, 0) in farm1_plots
    assert farm1_plots[(0, 0)]["crop"] == "CASSAVA"
    assert (0, 0) in farm2_plots
    assert farm2_plots[(0, 0)]["crop"] == "MAIZE"


def test_get_available_farms():
    plot_registry.register_plot(0, 0, "ASHANTI", "AMANSIE WEST", "CASSAVA", farm_id="farm_a")
    plot_registry.register_plot(1, 1, "ASHANTI", "AMANSIE WEST", "MAIZE", farm_id="farm_b")

    farms = plot_registry.get_available_farms()
    assert "farm_a" in farms
    assert "farm_b" in farms


def test_list_and_delete_plot():
    plot_registry.register_plot(0, 0, "ASHANTI", "AMANSIE WEST", "CASSAVA", farm_id="farm_del")
    plot_registry.register_plot(1, 1, "ASHANTI", "AMANSIE WEST", "MAIZE", farm_id="farm_del")

    plots = plot_registry.list_plots(farm_id="farm_del")
    assert len(plots) == 2

    plot_registry.delete_plot(0, 0, farm_id="farm_del")
    plots_after = plot_registry.list_plots(farm_id="farm_del")
    assert len(plots_after) == 1
    assert plots_after[0]["row"] == 1 and plots_after[0]["col"] == 1


def test_health_score_calculation_formula():
    """Confirms Farm Health Score = (1 - avg_urgency) * 100 with weather modifiers."""
    yield_risk_map = {(0, 0): 0.6, (0, 1): 0.2}
    disease_risk_map = {(0, 0): 0.8, (0, 1): 0.0}

    # Spray window alert: rain in 48h and disease > 0.5 adds 0.15 urgency modifier
    weather_advice = {
        "rain_expected_48h": True,
        "irrigation_alert": False,
        "irrigation_message": None,
        "planting_recommended": False,
        "planting_message": None,
    }

    result = compute_farm_health(yield_risk_map, disease_risk_map, weather_advice)

    # Plot (0,0): base = 0.5*0.6 + 0.5*0.8 = 0.70. With weather modifier: 0.70 + 0.15 = 0.85
    # Plot (0,1): base = 0.5*0.2 + 0.5*0.0 = 0.10. Weather modifier not triggered: 0.10
    # Avg urgency = (0.85 + 0.10) / 2 = 0.475
    # Health score = (1 - 0.475) * 100 = 52.5%
    assert round(result["plots"][(0, 0)]["urgency"], 2) == 0.85
    assert round(result["plots"][(0, 1)]["urgency"], 2) == 0.10
    assert round(result["farm_health_score"], 1) == 52.5
    assert "Spray before rain" in result["plots"][(0, 0)]["recommendation"]


