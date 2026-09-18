"""
PuLP-based linear programming optimizer for the GridWise 24-hour battery schedule.
Minimizes total grid electricity cost subject to energy balance, battery, solar,
and operator-directive constraints.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pulp

from models import (
    BatteryAction,
    BatterySpec,
    DirectiveInterpretation,
    DirectiveType,
    HourEntry,
    HourlyPlanEntry,
)

logger = logging.getLogger(__name__)


def optimize_schedule(
    hours: List[HourEntry],
    battery: BatterySpec,
    directives: List[DirectiveInterpretation],
) -> Dict[str, Any]:
    """
    Solve the 24-hour energy scheduling problem using PuLP.

    Returns a dict with:
      - hourly_plan: List[HourlyPlanEntry]
      - total_grid_kwh: float
      - total_cost_bdt: float
      - peak_grid_kwh: float
    """
    # Sort hours by hour index
    hours_sorted = sorted(hours, key=lambda h: h.hour)
    H = list(range(24))

    # Pre-compute effective solar after solar_reduction directives
    effective_solar = {h.hour: h.solar_kwh for h in hours_sorted}
    demand = {h.hour: h.demand_kwh for h in hours_sorted}
    tariff = {h.hour: h.tariff_bdt_per_kwh for h in hours_sorted}

    # Parse directives into constraint sets
    no_charge_hours = set()
    no_discharge_hours = set()
    min_reserve_hours: Dict[int, float] = {}  # hour -> minimum_energy_kwh
    max_grid_hours: Dict[int, float] = {}  # hour -> max_grid_kwh

    for d in directives:
        if not d.applies or d.directive_type == DirectiveType.no_op:
            continue

        adj = d.structured_adjustment
        if adj is None:
            continue

        d_hours = adj.get("hours", [])

        if d.directive_type == DirectiveType.solar_reduction:
            factor = adj.get("factor", 1.0)
            for h in d_hours:
                effective_solar[h] = effective_solar.get(h, 0) * factor

        elif d.directive_type == DirectiveType.no_charge_window:
            for h in d_hours:
                no_charge_hours.add(h)

        elif d.directive_type == DirectiveType.no_discharge_window:
            for h in d_hours:
                no_discharge_hours.add(h)

        elif d.directive_type == DirectiveType.minimum_battery_reserve:
            min_energy = adj.get("minimum_energy_kwh", 0)
            for h in d_hours:
                # Take the max if multiple reserves apply to same hour
                min_reserve_hours[h] = max(min_reserve_hours.get(h, 0), min_energy)

        elif d.directive_type == DirectiveType.max_grid_window:
            max_grid = adj.get("max_grid_kwh", float("inf"))
            for h in d_hours:
                # Take the min if multiple caps apply to same hour
                if h in max_grid_hours:
                    max_grid_hours[h] = min(max_grid_hours[h], max_grid)
                else:
                    max_grid_hours[h] = max_grid

    # ─── PuLP Model ──────────────────────────────────────────────────────
    prob = pulp.LpProblem("GridWise_MinCost", pulp.LpMinimize)

    # Decision variables
    grid = pulp.LpVariable.dicts("grid", H, lowBound=0)
    solar_used = pulp.LpVariable.dicts("solar_used", H, lowBound=0)
    charge = pulp.LpVariable.dicts("charge", H, lowBound=0)
    discharge = pulp.LpVariable.dicts("discharge", H, lowBound=0)
    battery_energy = pulp.LpVariable.dicts("battery_energy", H, lowBound=0)

    # Objective: minimize total grid cost
    prob += pulp.lpSum(grid[h] * tariff[h] for h in H), "TotalCost"

    for h in H:
        # ─── Energy balance ──────────────────────────────────────────
        # grid + solar_used + discharge = demand + charge
        prob += (
            grid[h] + solar_used[h] + discharge[h] == demand[h] + charge[h],
            f"EnergyBalance_{h}",
        )

        # ─── Solar limit ─────────────────────────────────────────────
        prob += (
            solar_used[h] <= effective_solar[h],
            f"SolarCap_{h}",
        )

        # ─── Battery state transition ────────────────────────────────
        if h == 0:
            prob += (
                battery_energy[h] == battery.initial_energy_kwh + charge[h] - discharge[h],
                f"BatteryState_{h}",
            )
        else:
            prob += (
                battery_energy[h] == battery_energy[h - 1] + charge[h] - discharge[h],
                f"BatteryState_{h}",
            )

        # ─── Battery capacity bounds ─────────────────────────────────
        # Effective minimum for this hour
        base_min = battery.minimum_energy_kwh
        eff_min = max(base_min, min_reserve_hours.get(h, 0))

        prob += (
            battery_energy[h] >= eff_min,
            f"BatteryMin_{h}",
        )
        prob += (
            battery_energy[h] <= battery.capacity_kwh,
            f"BatteryMax_{h}",
        )

        # ─── Charge/discharge rate limits ────────────────────────────
        prob += (
            charge[h] <= battery.max_charge_kwh_per_hour,
            f"ChargeRate_{h}",
        )
        prob += (
            discharge[h] <= battery.max_discharge_kwh_per_hour,
            f"DischargeRate_{h}",
        )

        # ─── No-charge constraint ────────────────────────────────────
        if h in no_charge_hours:
            prob += (charge[h] == 0, f"NoCharge_{h}")

        # ─── No-discharge constraint ─────────────────────────────────
        if h in no_discharge_hours:
            prob += (discharge[h] == 0, f"NoDischarge_{h}")

        # ─── Max grid constraint ─────────────────────────────────────
        if h in max_grid_hours:
            prob += (grid[h] <= max_grid_hours[h], f"MaxGrid_{h}")

    # ─── End-of-day battery neutrality ───────────────────────────────
    prob += (
        battery_energy[23] == battery.initial_energy_kwh,
        "EndOfDayNeutrality",
    )

    # ─── Solve ───────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=30)
    status = prob.solve(solver)

    if pulp.LpStatus[status] != "Optimal":
        logger.error("Solver status: %s", pulp.LpStatus[status])
        raise ValueError(f"Optimization failed: solver status = {pulp.LpStatus[status]}")

    # ─── Extract results ─────────────────────────────────────────────
    hourly_plan: List[HourlyPlanEntry] = []
    total_grid = 0.0
    total_cost = 0.0
    peak_grid = 0.0

    for h in H:
        g = round(pulp.value(grid[h]), 4)
        s = round(pulp.value(solar_used[h]), 4)
        ch = round(pulp.value(charge[h]), 4)
        dis = round(pulp.value(discharge[h]), 4)
        be = round(pulp.value(battery_energy[h]), 4)

        # Determine battery action
        # Use a small threshold to handle floating point noise
        EPSILON = 1e-6
        if ch > EPSILON and dis > EPSILON:
            # Both non-zero: shouldn't happen in optimal solution
            # but pick the dominant one
            if ch >= dis:
                action = BatteryAction.charge
                batt_kwh = round(ch - dis, 4)
                if batt_kwh < EPSILON:
                    action = BatteryAction.idle
                    batt_kwh = 0.0
            else:
                action = BatteryAction.discharge
                batt_kwh = round(dis - ch, 4)
                if batt_kwh < EPSILON:
                    action = BatteryAction.idle
                    batt_kwh = 0.0
        elif ch > EPSILON:
            action = BatteryAction.charge
            batt_kwh = ch
        elif dis > EPSILON:
            action = BatteryAction.discharge
            batt_kwh = dis
        else:
            action = BatteryAction.idle
            batt_kwh = 0.0

        # Round for clean output
        g = round(g, 2)
        s = round(s, 2)
        batt_kwh = round(batt_kwh, 2)
        be = round(be, 2)

        hourly_plan.append(HourlyPlanEntry(
            hour=h,
            grid_kwh=g,
            solar_used_kwh=s,
            battery_action=action,
            battery_kwh=batt_kwh,
            battery_energy_after_kwh=be,
        ))

        total_grid += g
        total_cost += g * tariff[h]
        peak_grid = max(peak_grid, g)

    return {
        "hourly_plan": hourly_plan,
        "total_grid_kwh": round(total_grid, 2),
        "total_cost_bdt": round(total_cost, 2),
        "peak_grid_kwh": round(peak_grid, 2),
    }
