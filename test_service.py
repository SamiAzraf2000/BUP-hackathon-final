"""
Service and integration tests for GridWise API (bup_solution).
Tests valid flow, 400 validation errors, and Safe Failure (Section 08).
"""

import copy
from unittest.mock import patch
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

BASE_PAYLOAD = {
    "scenario_id": "TEST-01",
    "operator_notes": [
        "Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, usable solar should be treated as roughly 25% of the forecast.",
        "The sports office moved next month's registration deadline."
    ],
    "hours": [
        {"hour": 0, "demand_kwh": 90, "solar_kwh": 0, "tariff_bdt_per_kwh": 6},
        {"hour": 1, "demand_kwh": 85, "solar_kwh": 0, "tariff_bdt_per_kwh": 6},
        {"hour": 2, "demand_kwh": 80, "solar_kwh": 0, "tariff_bdt_per_kwh": 5},
        {"hour": 3, "demand_kwh": 80, "solar_kwh": 0, "tariff_bdt_per_kwh": 5},
        {"hour": 4, "demand_kwh": 85, "solar_kwh": 0, "tariff_bdt_per_kwh": 5},
        {"hour": 5, "demand_kwh": 95, "solar_kwh": 0, "tariff_bdt_per_kwh": 6},
        {"hour": 6, "demand_kwh": 110, "solar_kwh": 5, "tariff_bdt_per_kwh": 8},
        {"hour": 7, "demand_kwh": 130, "solar_kwh": 20, "tariff_bdt_per_kwh": 10},
        {"hour": 8, "demand_kwh": 150, "solar_kwh": 50, "tariff_bdt_per_kwh": 12},
        {"hour": 9, "demand_kwh": 165, "solar_kwh": 90, "tariff_bdt_per_kwh": 14},
        {"hour": 10, "demand_kwh": 175, "solar_kwh": 130, "tariff_bdt_per_kwh": 16},
        {"hour": 11, "demand_kwh": 180, "solar_kwh": 160, "tariff_bdt_per_kwh": 16},
        {"hour": 12, "demand_kwh": 185, "solar_kwh": 180, "tariff_bdt_per_kwh": 15},
        {"hour": 13, "demand_kwh": 180, "solar_kwh": 170, "tariff_bdt_per_kwh": 14},
        {"hour": 14, "demand_kwh": 170, "solar_kwh": 140, "tariff_bdt_per_kwh": 13},
        {"hour": 15, "demand_kwh": 165, "solar_kwh": 90, "tariff_bdt_per_kwh": 14},
        {"hour": 16, "demand_kwh": 170, "solar_kwh": 45, "tariff_bdt_per_kwh": 18},
        {"hour": 17, "demand_kwh": 185, "solar_kwh": 10, "tariff_bdt_per_kwh": 22},
        {"hour": 18, "demand_kwh": 205, "solar_kwh": 0, "tariff_bdt_per_kwh": 28},
        {"hour": 19, "demand_kwh": 215, "solar_kwh": 0, "tariff_bdt_per_kwh": 30},
        {"hour": 20, "demand_kwh": 205, "solar_kwh": 0, "tariff_bdt_per_kwh": 26},
        {"hour": 21, "demand_kwh": 175, "solar_kwh": 0, "tariff_bdt_per_kwh": 18},
        {"hour": 22, "demand_kwh": 135, "solar_kwh": 0, "tariff_bdt_per_kwh": 10},
        {"hour": 23, "demand_kwh": 105, "solar_kwh": 0, "tariff_bdt_per_kwh": 7}
    ],
    "battery": {
        "capacity_kwh": 220,
        "initial_energy_kwh": 110,
        "minimum_energy_kwh": 40,
        "max_charge_kwh_per_hour": 50,
        "max_discharge_kwh_per_hour": 50
    }
}


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}
    print("PASS: test_health")


def test_validation_errors_return_400():
    # 1. Missing required field
    bad_payload = copy.deepcopy(BASE_PAYLOAD)
    del bad_payload["battery"]
    res = client.post("/optimize-energy", json=bad_payload)
    assert res.status_code == 400, f"Expected 400 for missing field, got {res.status_code}"

    # 2. Too few hours (23 instead of 24)
    bad_payload = copy.deepcopy(BASE_PAYLOAD)
    bad_payload["hours"] = bad_payload["hours"][:23]
    res = client.post("/optimize-energy", json=bad_payload)
    assert res.status_code == 400, f"Expected 400 for 23 hours, got {res.status_code}"

    # 3. Duplicate hours (hour 0 repeated, hour 23 missing)
    bad_payload = copy.deepcopy(BASE_PAYLOAD)
    bad_payload["hours"][23] = copy.deepcopy(bad_payload["hours"][0])
    res = client.post("/optimize-energy", json=bad_payload)
    assert res.status_code == 400, f"Expected 400 for duplicate hour, got {res.status_code}"

    # 4. More than 3 operator notes
    bad_payload = copy.deepcopy(BASE_PAYLOAD)
    bad_payload["operator_notes"] = ["Note 1", "Note 2", "Note 3", "Note 4"]
    res = client.post("/optimize-energy", json=bad_payload)
    assert res.status_code == 400, f"Expected 400 for >3 notes, got {res.status_code}"

    # 5. Invalid battery state (initial < minimum)
    bad_payload = copy.deepcopy(BASE_PAYLOAD)
    bad_payload["battery"]["initial_energy_kwh"] = 20
    bad_payload["battery"]["minimum_energy_kwh"] = 40
    res = client.post("/optimize-energy", json=bad_payload)
    assert res.status_code == 400, f"Expected 400 for initial < minimum energy, got {res.status_code}"

    print("PASS: test_validation_errors_return_400 (All 5 error conditions correctly return HTTP 400)")


def test_safe_failure_on_llm_exception():
    """Verify Section 08 compliance: Provider errors must NOT 500, must fall back safely to no_op."""
    from llm_interpreter import _CACHE
    _CACHE.clear()
    with patch("llm_interpreter.genai.GenerativeModel.generate_content", side_effect=Exception("Provider 503 Outage")):
        res = client.post("/optimize-energy", json=BASE_PAYLOAD)
        assert res.status_code == 200, f"Expected 200 on LLM failure fallback, got {res.status_code}"
        data = res.json()
        assert len(data["directive_interpretation"]) == 2
        for d in data["directive_interpretation"]:
            assert d["directive_type"] == "no_op"
            assert d["applies"] is False
            assert d["structured_adjustment"] is None
        assert len(data["hourly_plan"]) == 24
        print("PASS: test_safe_failure_on_llm_exception (Safe failure succeeded: 0% 500 errors)")


if __name__ == "__main__":
    test_health()
    test_validation_errors_return_400()
    test_safe_failure_on_llm_exception()
    print("\nAll integration & safety tests passed successfully!")
