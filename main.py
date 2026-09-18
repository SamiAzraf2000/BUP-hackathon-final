"""
GridWise Smart Campus Energy Optimization API.
FastAPI service for BUP CSE Fest 2026 Hackathon Preliminary.
"""

from __future__ import annotations

import logging
import os
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from llm_interpreter import interpret_notes
from models import (
    OptimizationRequest,
    OptimizationResponse,
)
from optimizer import optimize_schedule

# Load environment variables from .env file if present
load_dotenv()

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="GridWise Smart Campus Energy Optimization",
    description="LLM-assisted 24-hour energy scheduling for the BUP CSE Fest 2026 Hackathon.",
    version="1.0.0",
)


# ─── Health Endpoint ─────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Return service readiness status."""
    return {"status": "ok"}


# ─── Optimize-Energy Endpoint ────────────────────────────────────────────────

@app.post("/optimize-energy", response_model=OptimizationResponse)
async def optimize_energy(request: OptimizationRequest):
    """
    Accept one scenario JSON object, interpret operator notes via LLM,
    apply directives, optimize the 24-hour battery schedule, and return
    the interpretation + optimization plan.
    """
    try:
        logger.info("Processing scenario: %s", request.scenario_id)

        # Validate hours cover 0-23
        hour_set = {h.hour for h in request.hours}
        if hour_set != set(range(24)):
            raise HTTPException(
                status_code=400,
                detail="hours array must contain exactly 24 entries for hours 0 through 23.",
            )

        # Validate battery constraints are self-consistent
        if request.battery.initial_energy_kwh > request.battery.capacity_kwh:
            raise HTTPException(
                status_code=400,
                detail="initial_energy_kwh cannot exceed capacity_kwh.",
            )
        if request.battery.minimum_energy_kwh > request.battery.capacity_kwh:
            raise HTTPException(
                status_code=400,
                detail="minimum_energy_kwh cannot exceed capacity_kwh.",
            )

        # ── Step 1: LLM Interpretation ───────────────────────────────────
        logger.info("Interpreting %d operator notes via LLM...", len(request.operator_notes))
        directive_interpretations = await interpret_notes(
            operator_notes=request.operator_notes,
            battery_capacity_kwh=request.battery.capacity_kwh,
        )
        logger.info("LLM interpretation complete: %d directives", len(directive_interpretations))

        # ── Step 2: Optimize Schedule ────────────────────────────────────
        logger.info("Running optimization...")
        result = optimize_schedule(
            hours=request.hours,
            battery=request.battery,
            directives=directive_interpretations,
        )
        logger.info(
            "Optimization complete: cost=%.2f BDT, grid=%.2f kWh",
            result["total_cost_bdt"],
            result["total_grid_kwh"],
        )

        # ── Step 3: Build Response ───────────────────────────────────────
        # Generate a concise plan summary
        active_directives = [
            d for d in directive_interpretations if d.applies
        ]
        no_ops = [
            d for d in directive_interpretations if not d.applies
        ]

        summary_parts = []
        for d in active_directives:
            summary_parts.append(f"Applied {d.directive_type.value} directive.")
        if no_ops:
            summary_parts.append(f"Ignored {len(no_ops)} non-actionable note(s).")
        summary_parts.append(
            f"Total cost: {result['total_cost_bdt']:.2f} BDT. "
            f"Peak grid: {result['peak_grid_kwh']:.2f} kWh."
        )
        plan_summary = " ".join(summary_parts)

        response = OptimizationResponse(
            scenario_id=request.scenario_id,
            directive_interpretation=directive_interpretations,
            hourly_plan=result["hourly_plan"],
            total_grid_kwh=result["total_grid_kwh"],
            total_cost_bdt=result["total_cost_bdt"],
            peak_grid_kwh=result["peak_grid_kwh"],
            plan_summary=plan_summary,
        )

        return response

    except HTTPException:
        raise
    except ValueError as e:
        logger.error("Optimization error: %s", e)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Internal error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error.")


# ─── Generic error handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )


# ─── Run with uvicorn ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
