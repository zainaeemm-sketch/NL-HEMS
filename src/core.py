from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from ortools.sat.python import cp_model


# =========================
# Schemas
# =========================
class TimeWindow(BaseModel):
    start_hour: int = Field(ge=0, le=23)
    end_hour: int = Field(ge=1, le=24)


class ComfortPreference(BaseModel):
    thermal_label: str
    comfort_priority: float = Field(ge=0.0, le=1.0)


class CostPreference(BaseModel):
    cost_sensitivity_label: str
    cost_priority: float = Field(ge=0.0, le=1.0)
    soft_budget_eur: Optional[float] = None


class DRPreference(BaseModel):
    participation_willingness: str
    dr_priority: float = Field(ge=0.0, le=1.0)


class ParsedIntent(BaseModel):
    raw_command: str
    guest_event: bool = False
    guest_arrival_hour: Optional[int] = None
    time_window: TimeWindow
    comfort: ComfortPreference
    cost: CostPreference
    dr: DRPreference
    clarification_needed: bool = False
    notes: List[str] = Field(default_factory=list)


# =========================
# Parser Stub (baseline)
# =========================
def parse_command_stub(command: str) -> ParsedIntent:
    text = command.lower()

    guest_event = "guest" in text or "guests" in text
    warm = "warm" in text or "cozy" in text
    save_money = (
        "don't run up the bill" in text
        or "save" in text
        or "cheap" in text
        or "expensive" in text
    )

    if "tonight" in text:
        tw = TimeWindow(start_hour=18, end_hour=23)
    elif "evening" in text:
        tw = TimeWindow(start_hour=17, end_hour=22)
    else:
        tw = TimeWindow(start_hour=17, end_hour=22)

    comfort_priority = 0.8 if (guest_event and warm) else (0.6 if warm else 0.5)
    cost_priority = 0.8 if save_money else 0.4

    notes = []
    if guest_event and save_money:
        notes.append("Potential comfort-cost conflict during evening peak.")

    return ParsedIntent(
        raw_command=command,
        guest_event=guest_event,
        guest_arrival_hour=19 if guest_event else None,
        time_window=tw,
        comfort=ComfortPreference(
            thermal_label="warm" if warm else "neutral",
            comfort_priority=comfort_priority,
        ),
        cost=CostPreference(
            cost_sensitivity_label="high" if save_money else "medium",
            cost_priority=cost_priority,
            soft_budget_eur=7.0 if save_money else None,
        ),
        dr=DRPreference(
            participation_willingness="conditional",
            dr_priority=0.6,
        ),
        clarification_needed=False,
        notes=notes,
    )


# =========================
# Fuzzy mapping
# =========================
def map_fuzzy_preferences(intent: ParsedIntent, enforce_guest_hard_comfort: bool = False) -> dict:
    if intent.comfort.thermal_label == "warm":
        target_temp_c, min_temp_c = 22.0, 21.0
    else:
        target_temp_c, min_temp_c = 21.0, 20.0

    return {
        "target_temp_c": target_temp_c,
        "min_temp_c": min_temp_c,
        "comfort_weight": max(1, int(10 * intent.comfort.comfort_priority)),
        "cost_weight": 8 if intent.cost.cost_sensitivity_label == "high" else 4,
        "dr_weight": max(1, int(10 * intent.dr.dr_priority)),
        "time_window": {"start_hour": intent.time_window.start_hour, "end_hour": intent.time_window.end_hour},
        "guest_event": intent.guest_event,
        "enforce_guest_hard_comfort": bool(enforce_guest_hard_comfort),
    }


# =========================
# Optimizer (NOW CONTEXT-AWARE)
# =========================
def optimize_schedule(mapped: dict, context: Dict[str, Any] | None = None):
    """
    Constraint optimization (OR-Tools CP-SAT).
    Decision: HVAC on/off per hour.
    Objective: cost + DR penalty + discomfort + (optional) guest target + switching penalty.

    context overrides (all optional):
      - horizon_start_hour (int, default 16)
      - horizon_end_hour (int, default 24)  # exclusive
      - price (dict {hour:int price_level})
      - dr_event_hours (list[int])
      - initial_temp_c (float)
      - loss_c_per_hour (float) OR loss (int scaled-by-10)
      - heat_c_per_hour (float) OR heat (int scaled-by-10)
      - cost_weight_scale, comfort_weight_scale, dr_weight_scale (float)
      - dr_penalty_per_on (int, default 5)
      - switch_penalty (int, default 2)
    """
    context = context or {}

    # Horizon
    h_start = int(context.get("horizon_start_hour", 16))
    h_end = int(context.get("horizon_end_hour", 24))
    if h_end <= h_start:
        h_end = h_start + 1
    hours = list(range(h_start, h_end))
    n = len(hours)

    # Default price profile (can be overridden)
    default_price = {h: 2 for h in hours}
    for h in hours:
        if h == 17:
            default_price[h] = 3
        if h in (18, 20):
            default_price[h] = 8
        if h == 19:
            default_price[h] = 9
        if h == 21:
            default_price[h] = 5
        if h == 22:
            default_price[h] = 3
        if h == 23:
            default_price[h] = 2

    price_in = context.get("price", default_price)
    # JSON often stores keys as strings; normalize
    price = {int(k): int(v) for k, v in price_in.items()}

    # DR hours
    dr_event_hours = set(int(x) for x in context.get("dr_event_hours", [18, 19]))
    dr_event_hours = set(h for h in dr_event_hours if h in hours)

    # Thermal dynamics
    initial_temp_c = float(context.get("initial_temp_c", 20.0))

    # Support both: loss (scaled by 10) OR loss_c_per_hour
    if "loss" in context:
        loss = int(context["loss"])
    else:
        loss = int(round(float(context.get("loss_c_per_hour", 0.4)) * 10))

    if "heat" in context:
        heat = int(context["heat"])
    else:
        heat = int(round(float(context.get("heat_c_per_hour", 0.8)) * 10))

    # Weight scaling (lets you “edit inputs” without changing parser)
    cost_scale = float(context.get("cost_weight_scale", 1.0))
    comfort_scale = float(context.get("comfort_weight_scale", 1.0))
    dr_scale = float(context.get("dr_weight_scale", 1.0))

    cost_weight = max(1, int(round(mapped["cost_weight"] * cost_scale)))
    comfort_weight = max(1, int(round(mapped["comfort_weight"] * comfort_scale)))
    dr_weight = max(1, int(round(mapped["dr_weight"] * dr_scale)))

    dr_penalty_per_on = int(context.get("dr_penalty_per_on", 5))
    switch_penalty = int(context.get("switch_penalty", 2))

    # Comfort parameters
    target_temp = int(mapped["target_temp_c"] * 10)
    min_temp = int(mapped["min_temp_c"] * 10)
    guest_event = bool(mapped.get("guest_event", False))
    tw_start = int(mapped["time_window"]["start_hour"])
    tw_end = int(mapped["time_window"]["end_hour"])
    enforce_guest_hard_comfort = bool(mapped.get("enforce_guest_hard_comfort", False))

    # --- Build CP-SAT model ---
    model = cp_model.CpModel()

    hvac_on = [model.NewBoolVar(f"hvac_on_{h}") for h in hours]
    temp = [model.NewIntVar(150, 300, f"temp_{i}") for i in range(n + 1)]
    discomfort = [model.NewIntVar(0, 200, f"discomfort_{i}") for i in range(n)]

    model.Add(temp[0] == int(round(initial_temp_c * 10)))

    for i, h in enumerate(hours):
        # thermal dynamics
        model.Add(temp[i + 1] == temp[i] - loss + heat * hvac_on[i])

        # discomfort below minimum
        model.Add(discomfort[i] >= min_temp - temp[i + 1])
        model.Add(discomfort[i] >= 0)

        # Optional hard guest comfort constraint in guest window
        if guest_event and enforce_guest_hard_comfort and (tw_start <= h < tw_end):
            model.Add(temp[i + 1] >= min_temp)

    # Soft guest target penalty
    extra_penalties = []
    if guest_event:
        for i, h in enumerate(hours):
            if tw_start <= h < tw_end:
                extra = model.NewIntVar(0, 200, f"guest_gap_{h}")
                model.Add(extra >= target_temp - temp[i + 1])
                model.Add(extra >= 0)
                extra_penalties.append(extra)

    # Switching penalty
    switch_vars = []
    for i in range(1, n):
        sw = model.NewIntVar(0, 1, f"switch_{i}")
        model.Add(sw >= hvac_on[i] - hvac_on[i - 1])
        model.Add(sw >= hvac_on[i - 1] - hvac_on[i])
        switch_vars.append(sw)

    # Objective
    objective_terms = []

    for i, h in enumerate(hours):
        objective_terms.append(cost_weight * price.get(h, 2) * hvac_on[i])
        if h in dr_event_hours:
            objective_terms.append(dr_weight * dr_penalty_per_on * hvac_on[i])

    for d in discomfort:
        objective_terms.append(comfort_weight * d)

    for e in extra_penalties:
        objective_terms.append(max(1, comfort_weight // 2) * e)

    for sw in switch_vars:
        objective_terms.append(switch_penalty * sw)

    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": "infeasible", "schedule": []}

    schedule = []
    for i, h in enumerate(hours):
        schedule.append({
            "hour": h,
            "hvac_on": int(solver.Value(hvac_on[i])),
            "temp_c": solver.Value(temp[i + 1]) / 10.0,
            "price_level": price.get(h, 2),
            "dr_event": h in dr_event_hours,
            "guest_window": bool(guest_event and (tw_start <= h < tw_end)),
        })

    return {
        "status": "ok",
        "schedule": schedule,
        "objective_value": solver.ObjectiveValue(),
        "context_used": {
            "horizon_start_hour": h_start,
            "horizon_end_hour": h_end,
            "initial_temp_c": initial_temp_c,
            "loss": loss,
            "heat": heat,
            "dr_event_hours": sorted(list(dr_event_hours)),
            "cost_weight": cost_weight,
            "comfort_weight": comfort_weight,
            "dr_weight": dr_weight,
        }
    }


# =========================
# Metrics
# =========================
def comfort_violation_minutes(schedule, min_temp_c=21.0):
    return sum(60 for row in schedule if row["temp_c"] < min_temp_c)


def dr_compliance_score(schedule):
    dr_rows = [r for r in schedule if r["dr_event"]]
    if not dr_rows:
        return 1.0
    return sum(1 for r in dr_rows if r["hvac_on"] == 0) / len(dr_rows)