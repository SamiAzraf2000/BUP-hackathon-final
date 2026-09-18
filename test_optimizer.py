"""
Test the PuLP optimizer against public sample cases.
This test bypasses the LLM and feeds the expected directive interpretations
directly to the optimizer, verifying it produces the correct optimal cost.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from models import (
    BatterySpec,
    DirectiveInterpretation,
    DirectiveType,
    HourEntry,
)
from optimizer import optimize_schedule


def load_samples():
    """Load the public sample cases JSON."""
    path = os.path.join(os.path.dirname(__file__), "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


def build_directives(expected_interp: list) -> list:
    """Convert expected_output directive_interpretation dicts to model objects."""
    directives = []
    for d in expected_interp:
        directives.append(DirectiveInterpretation(
            note_index=d["note_index"],
            applies=d["applies"],
            directive_type=DirectiveType(d["directive_type"]),
            structured_adjustment=d["structured_adjustment"],
            explanation=d.get("explanation", ""),
        ))
    return directives


def run_sample(case: dict) -> bool:
    """Test a single sample case. Returns True if cost matches within tolerance."""
    case_id = case["id"]
    inp = case["input"]
    expected = case["expected_output"]

    hours = [HourEntry(**h) for h in inp["hours"]]
    battery = BatterySpec(**inp["battery"])
    directives = build_directives(expected["directive_interpretation"])

    try:
        result = optimize_schedule(hours, battery, directives)
    except Exception as e:
        print(f"  FAIL {case_id}: Optimizer failed: {e}")
        return False

    expected_cost = expected["total_cost_bdt"]
    actual_cost = result["total_cost_bdt"]
    cost_diff = abs(actual_cost - expected_cost)

    expected_grid = expected["total_grid_kwh"]
    actual_grid = result["total_grid_kwh"]
    grid_diff = abs(actual_grid - expected_grid)

    # Validate energy balance for each hour
    hours_map = {h.hour: h for h in hours}
    
    # Apply solar reductions to get effective solar
    effective_solar = {h.hour: h.solar_kwh for h in hours}
    for d in directives:
        if d.applies and d.directive_type == DirectiveType.solar_reduction:
            adj = d.structured_adjustment
            for hr in adj["hours"]:
                effective_solar[hr] = effective_solar[hr] * adj["factor"]

    balance_ok = True
    for entry in result["hourly_plan"]:
        h = entry.hour
        h_data = hours_map[h]
        
        # Energy balance: grid + solar + discharge = demand + charge
        charge_kwh = entry.battery_kwh if entry.battery_action.value == "charge" else 0
        discharge_kwh = entry.battery_kwh if entry.battery_action.value == "discharge" else 0
        
        lhs = entry.grid_kwh + entry.solar_used_kwh + discharge_kwh
        rhs = h_data.demand_kwh + charge_kwh
        
        if abs(lhs - rhs) > 0.02:
            print(f"  WARN Hour {h}: Energy balance violated: {lhs:.2f} != {rhs:.2f}")
            balance_ok = False
        
        # Solar cap
        if entry.solar_used_kwh > effective_solar[h] + 0.01:
            print(f"  WARN Hour {h}: Solar usage {entry.solar_used_kwh:.2f} > effective {effective_solar[h]:.2f}")
            balance_ok = False

    # Check end-of-day neutrality
    final_battery = result["hourly_plan"][-1].battery_energy_after_kwh
    if abs(final_battery - battery.initial_energy_kwh) > 0.01:
        print(f"  WARN End-of-day battery: {final_battery:.2f} != {battery.initial_energy_kwh:.2f}")
        balance_ok = False

    passed = cost_diff <= 0.01 and balance_ok
    status = "PASS" if passed else "FAIL"
    print(f"  {status} {case_id}: cost={actual_cost:.2f} (expected {expected_cost:.2f}, diff={cost_diff:.2f}), "
          f"grid={actual_grid:.2f} (expected {expected_grid:.2f}, diff={grid_diff:.2f})")

    if not passed and cost_diff > 0.01:
        print(f"      Cost difference {cost_diff:.2f} exceeds tolerance of 0.01")

    return passed


def main():
    cases = load_samples()
    print(f"\nTesting {len(cases)} public sample cases (optimizer only, no LLM):\n")

    passed = 0
    failed = 0

    for case in cases:
        if run_sample(case):
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{passed+failed} passed")
    if failed > 0:
        print(f"WARNING: {failed} case(s) failed!")
        sys.exit(1)
    else:
        print("All cases passed!")


if __name__ == "__main__":
    main()
