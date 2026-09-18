"""
LLM Interpreter for operator notes.
Uses Google Gemini to extract structured directives from natural-language operator notes.
Includes deterministic guardrails as required by the Problem Statement (Section 08).
"""

from __future__ import annotations

import json
import os
import logging
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from models import (
    DirectiveInterpretation,
    DirectiveType,
)

logger = logging.getLogger(__name__)

# ─── Gemini Configuration ───────────────────────────────────────────────────

_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

_SYSTEM_PROMPT = """You are an expert energy-grid operator assistant. You will be given 1-3 natural-language operator notes about a campus energy system for a single 24-hour day.

For EACH note, you must classify it into exactly ONE of these directive types:

1. **solar_reduction** — The note says usable solar energy is reduced during specific hours.
   - structured_adjustment: {"hours": [list of integer hours 0-23], "factor": <float 0-1>}
   - "factor" is the FRACTION OF SOLAR THAT REMAINS (not the reduction). 
   - An "80% reduction" means factor = 0.2 (20% remains).
   - "roughly 25% of forecast" means factor = 0.25.
   - "one-fifth" means factor = 0.2.

2. **minimum_battery_reserve** — The note requires keeping battery energy at or above a minimum level during specific hours.
   - structured_adjustment: {"hours": [list of integer hours 0-23], "minimum_energy_kwh": <number>}
   - If the note says "50% of capacity" and battery capacity is provided in context, compute the absolute kWh value.

3. **no_charge_window** — The note says battery charging is unavailable during specific hours.
   - structured_adjustment: {"hours": [list of integer hours 0-23]}

4. **no_discharge_window** — The note says battery discharging is unavailable during specific hours.
   - structured_adjustment: {"hours": [list of integer hours 0-23]}

5. **max_grid_window** — The note limits grid electricity import to a maximum amount during specific hours.
   - structured_adjustment: {"hours": [list of integer hours 0-23], "max_grid_kwh": <number>}

6. **no_op** — The note is irrelevant to the 24-hour energy schedule (e.g., administrative updates, future dates, unrelated campus info).

CRITICAL TIME RULES:
- Time windows are START-INCLUSIVE, END-EXCLUSIVE using whole hours.
- "noon until 2 PM" = hours [12, 13]
- "1 PM to 3 PM" = hours [13, 14]
- "6 PM until 9 PM" = hours [18, 19, 20]
- "2 AM until 5 AM" = hours [2, 3, 4]
- "from midnight to 4 AM" = hours [0, 1, 2, 3]
- Hours must be unique integers 0-23 in ascending order.

CRITICAL RULES:
- Return EXACTLY one interpretation per note.
- For no_op: applies = false, structured_adjustment = null.
- For ALL other directives: applies = true.
- Do not invent directives not supported above.
- Do not modify demand, tariff, or battery parameters.

Respond with a JSON array of objects, one per note, in order. Each object has:
{
  "note_index": <int, 0-based>,
  "applies": <bool>,
  "directive_type": "<string>",
  "structured_adjustment": <object or null>,
  "explanation": "<short string>"
}
"""


def _configure_genai() -> None:
    """Configure the Gemini API with the environment key."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please set it to your Google Gemini API key."
        )
    genai.configure(api_key=api_key)


def _build_user_prompt(
    operator_notes: List[str],
    battery_capacity_kwh: float,
) -> str:
    """Build the user prompt with notes and battery context."""
    lines = [
        f"Battery capacity: {battery_capacity_kwh} kWh",
        "",
        "Operator notes:",
    ]
    for i, note in enumerate(operator_notes):
        lines.append(f"  [{i}] \"{note}\"")
    lines.append("")
    lines.append(
        "Return a JSON array with exactly one interpretation object per note, "
        "in note_index order. Only output the JSON array, nothing else."
    )
    return "\n".join(lines)


async def interpret_notes(
    operator_notes: List[str],
    battery_capacity_kwh: float,
) -> List[DirectiveInterpretation]:
    """
    Use Gemini to interpret operator notes into structured directives.
    Applies deterministic guardrails after LLM output.
    """
    _configure_genai()

    model = genai.GenerativeModel(
        model_name=_MODEL_NAME,
        system_instruction=_SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )

    user_prompt = _build_user_prompt(operator_notes, battery_capacity_kwh)
    logger.info("Sending %d notes to LLM for interpretation", len(operator_notes))

    response = model.generate_content(user_prompt)
    raw_text = response.text.strip()
    logger.debug("LLM raw response: %s", raw_text)

    # Parse the JSON array from LLM output
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error("LLM returned invalid JSON: %s", e)
        # Fallback: treat all notes as no_op
        return _fallback_all_no_op(operator_notes)

    if not isinstance(parsed, list):
        logger.error("LLM did not return a JSON array")
        return _fallback_all_no_op(operator_notes)

    # Apply deterministic guardrails
    validated = _apply_guardrails(parsed, operator_notes, battery_capacity_kwh)
    return validated


def _fallback_all_no_op(operator_notes: List[str]) -> List[DirectiveInterpretation]:
    """Fallback: mark all notes as no_op if LLM fails."""
    return [
        DirectiveInterpretation(
            note_index=i,
            applies=False,
            directive_type=DirectiveType.no_op,
            structured_adjustment=None,
            explanation="LLM interpretation failed; treating as non-actionable.",
        )
        for i in range(len(operator_notes))
    ]


def _apply_guardrails(
    parsed: List[dict],
    operator_notes: List[str],
    battery_capacity_kwh: float,
) -> List[DirectiveInterpretation]:
    """
    Deterministic guardrails applied after LLM output (Section 08).
    Validates structure, types, hours, and numeric ranges.
    """
    valid_types = {e.value for e in DirectiveType}
    results: List[DirectiveInterpretation] = []
    seen_indices = set()

    # Ensure we have one entry per note
    if len(parsed) != len(operator_notes):
        logger.warning(
            "LLM returned %d entries but expected %d; padding/truncating",
            len(parsed), len(operator_notes),
        )

    for i in range(len(operator_notes)):
        # Find the entry for this note_index
        entry = None
        for p in parsed:
            if isinstance(p, dict) and p.get("note_index") == i:
                entry = p
                break

        if entry is None:
            # Missing entry -> no_op fallback
            results.append(DirectiveInterpretation(
                note_index=i,
                applies=False,
                directive_type=DirectiveType.no_op,
                structured_adjustment=None,
                explanation="No LLM interpretation available for this note.",
            ))
            continue

        seen_indices.add(i)
        directive_type_str = entry.get("directive_type", "no_op")

        # Validate directive type
        if directive_type_str not in valid_types:
            logger.warning("Invalid directive type '%s' from LLM, using no_op", directive_type_str)
            directive_type_str = "no_op"

        directive_type = DirectiveType(directive_type_str)
        adj = entry.get("structured_adjustment")
        explanation = entry.get("explanation", "")

        if directive_type == DirectiveType.no_op:
            results.append(DirectiveInterpretation(
                note_index=i,
                applies=False,
                directive_type=DirectiveType.no_op,
                structured_adjustment=None,
                explanation=explanation or "This note does not affect today's energy schedule.",
            ))
            continue

        # Validate structured_adjustment for non-no_op types
        validated_adj = _validate_adjustment(directive_type, adj, battery_capacity_kwh)
        if validated_adj is None:
            # Validation failed -> fall back to no_op
            logger.warning("Adjustment validation failed for note %d, falling back to no_op", i)
            results.append(DirectiveInterpretation(
                note_index=i,
                applies=False,
                directive_type=DirectiveType.no_op,
                structured_adjustment=None,
                explanation="Structured adjustment validation failed; treating as non-actionable.",
            ))
            continue

        results.append(DirectiveInterpretation(
            note_index=i,
            applies=True,
            directive_type=directive_type,
            structured_adjustment=validated_adj,
            explanation=explanation,
        ))

    return results


def _validate_adjustment(
    directive_type: DirectiveType,
    adj: Optional[dict],
    battery_capacity_kwh: float,
) -> Optional[dict]:
    """
    Validate and sanitize a structured_adjustment dict.
    Returns the cleaned dict, or None if validation fails.
    """
    if adj is None or not isinstance(adj, dict):
        return None

    # All directives except no_op require an "hours" array
    hours = adj.get("hours")
    if not isinstance(hours, list) or len(hours) == 0:
        return None

    # Validate hours: unique integers 0-23 in ascending order
    try:
        hours = [int(h) for h in hours]
    except (ValueError, TypeError):
        return None

    if not all(0 <= h <= 23 for h in hours):
        return None

    hours = sorted(set(hours))

    if directive_type == DirectiveType.solar_reduction:
        factor = adj.get("factor")
        if factor is None:
            return None
        try:
            factor = float(factor)
        except (ValueError, TypeError):
            return None
        if not (0.0 <= factor <= 1.0):
            return None
        return {"hours": hours, "factor": round(factor, 4)}

    elif directive_type == DirectiveType.minimum_battery_reserve:
        min_energy = adj.get("minimum_energy_kwh")
        if min_energy is None:
            return None
        try:
            min_energy = float(min_energy)
        except (ValueError, TypeError):
            return None
        if min_energy < 0 or min_energy > battery_capacity_kwh:
            return None
        return {"hours": hours, "minimum_energy_kwh": round(min_energy, 2)}

    elif directive_type == DirectiveType.no_charge_window:
        return {"hours": hours}

    elif directive_type == DirectiveType.no_discharge_window:
        return {"hours": hours}

    elif directive_type == DirectiveType.max_grid_window:
        max_grid = adj.get("max_grid_kwh")
        if max_grid is None:
            return None
        try:
            max_grid = float(max_grid)
        except (ValueError, TypeError):
            return None
        if max_grid < 0:
            return None
        return {"hours": hours, "max_grid_kwh": round(max_grid, 2)}

    return None
