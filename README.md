# GridWise Smart Campus Energy Optimization

**BUP CSE Fest 2026 — Hackathon Preliminary**

An LLM-assisted energy scheduling API that interprets natural-language operator notes into structured directives, then uses mathematical optimization (linear programming) to produce the lowest-cost valid 24-hour battery schedule.

---

## Architecture Overview

```
Operator Notes (natural language)
        │
        ▼
┌───────────────────┐
│  Gemini LLM       │  Interprets notes → structured directives
│  (gemini-2.0-flash)│
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Deterministic    │  Validates types, hours, numeric ranges
│  Guardrails       │  (Section 08 compliance)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  PuLP LP Optimizer│  Minimizes grid cost subject to all constraints
│  (CBC solver)     │
└────────┬──────────┘
         │
         ▼
  24-hour optimal schedule + interpretation
```

## LLM Usage

The Google Gemini API (`gemini-2.0-flash`) is used **exclusively for interpreting operator notes** into one of the six supported directive types (`solar_reduction`, `minimum_battery_reserve`, `no_charge_window`, `no_discharge_window`, `max_grid_window`, `no_op`).

The LLM output is treated as untrusted data and passes through deterministic guardrails that validate:
- Directive type is one of the six supported types
- Hours are unique integers 0-23 in ascending order
- Numeric values (factor, reserve, grid cap) are within valid ranges
- Each note produces exactly one interpretation entry

## Optimizer / Solver

**PuLP** with the **CBC (COIN-OR Branch and Cut)** solver performs the cost minimization. The LP formulation includes:
- Energy balance constraints for all 24 hours
- Battery state transitions, capacity bounds, and rate limits
- Solar usage caps (with directive-modified effective solar)
- Directive-specific constraints (no-charge, no-discharge, grid cap, reserve)
- End-of-day battery neutrality

---

## Quick Start (Local)

### Prerequisites
- Python 3.10+
- A Google Gemini API key

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-dir>

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your-google-gemini-api-key-here
GEMINI_MODEL=gemini-2.0-flash
LOG_LEVEL=INFO
PORT=8000
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | — | Google Gemini API key |
| `GEMINI_MODEL` | No | `gemini-2.0-flash` | Gemini model to use |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `PORT` | No | `8000` | Server port |

### Run

```bash
# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000

# Or directly:
python main.py
```

### Test Health

```bash
curl http://localhost:8000/health
# Expected: {"status":"ok"}
```

### Test Optimize-Energy

```bash
curl -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_id": "TEST-01",
    "operator_notes": [
      "Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, usable solar should be treated as roughly 25% of the forecast.",
      "The sports office moved next month'\''s registration deadline."
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
  }'
```

---

## Automated Tests (Verification)

The project includes an exhaustive test suite to verify the optimizer's constraints (using the 10 public sample cases) and to guarantee API reliability (checking the schema validation and safe LLM fallback).

### Run Optimizer Tests

This verifies the linear programming logic against the 10 public sample cases without making network calls to Gemini. It tests if the energy balance, rate limits, and battery constraints perfectly match the rubric.

```bash
python test_optimizer.py
```
**Expected Output:** `10/10 passed`

### Run API & Integration Tests

This runs `test_service.py` to ensure all invalid requests are properly handled as `400 Bad Request` and that the application safely falls back to a `no_op` if the Gemini API goes down.

```bash
pytest test_service.py -v
```
**Expected Output:** `3 passed` (Testing health, 400 validation handlers, and Safe LLM Fallback).

---

## Docker Fallback

### Build

```bash
docker build -t gridwise:latest .
```

### Run

```bash
docker run -d \
  -p 8000:8000 \
  -e GEMINI_API_KEY=your-key-here \
  --name gridwise \
  gridwise:latest
```

### Verify

```bash
curl http://localhost:8000/health
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework for the API endpoints |
| `uvicorn` | ASGI server |
| `pydantic` | Request/response schema validation |
| `pulp` | Linear programming solver for cost minimization |
| `google-generativeai` | Google Gemini API client for note interpretation |
| `python-dotenv` | Environment variable management |
| `pytest` | Testing framework |
| `httpx` | Required by FastAPI for TestClient |

---

## Known Limitations

- The LLM interpretation depends on Gemini API availability and may introduce latency.
- If the Gemini API is unreachable, all notes will be treated as `no_op` (safe fallback).
- The optimizer uses the CBC solver bundled with PuLP; for very large-scale problems, a commercial solver may be faster.
- No secrets are committed to the repository; the `GEMINI_API_KEY` must be provided at runtime.

---

## Project Structure

```
├── main.py               # FastAPI application entry point
├── models.py             # Pydantic request/response models
├── llm_interpreter.py    # Gemini LLM note interpretation + guardrails
├── optimizer.py          # PuLP LP optimizer
├── test_optimizer.py     # LP exact constraint tests (10/10)
├── test_service.py       # API safety and 400 validation tests
├── requirements.txt      # Python dependencies
├── Dockerfile            # Production Docker image
├── .env                  # Environment variables (not committed)
└── README.md             # This file
```
