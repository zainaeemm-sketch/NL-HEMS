"""
src/metrics_extended.py
========================
Journal-version metrics: SCR, CVaR_α, Robust Feasibility Rate, plus Pareto helpers.

The in-sample versions of SCR and CVaR are already computed inside
`core_stochastic.optimize_schedule_stochastic`. This module provides:

  1. Out-of-sample evaluation: replay a fixed first-stage schedule
     {y(t), u_bat(t)} against a *separate* test scenario set to compute
     the Robust Feasibility Rate (RFR) — the standard journal-grade
     stress test.
  2. Pareto-frontier sweep: vary λ_comf / λ_cost to trace the comfort–cost
     frontier per parser × policy configuration. The journal needs this
     instead of the single scatter plot in conference Fig. 3.

Author: NL-HEMS journal extension
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional


# --------------------------------------------------------------------------- #
# CVaR helper (standalone, in case you want to recompute from raw violations)
# --------------------------------------------------------------------------- #

def cvar(values: List[float], probs: List[float], alpha: float = 0.2) -> float:
    """
    Compute CVaR_α of a discrete distribution: the expected value conditional
    on being in the worst α-fraction. `values` are loss values (e.g. comfort
    violation minutes); higher = worse.
    """
    if not values:
        return 0.0
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in (0, 1]")
    pairs = sorted(zip(values, probs), key=lambda x: -x[0])  # worst first
    cum_p = 0.0
    acc = 0.0
    for v, p in pairs:
        if cum_p >= alpha:
            break
        take = min(p, alpha - cum_p)
        acc += take * v
        cum_p += take
    return acc / alpha if alpha > 0 else 0.0


# --------------------------------------------------------------------------- #
# Out-of-sample replay
# --------------------------------------------------------------------------- #

def replay_first_stage(
    first_stage_schedule: List[Dict],
    baseline_context: Dict,
    test_scenarios: List,
    mapped: Dict,
    thermal_params=None,
    pv_battery_params=None,
    comfort_buffer_c: float = 0.0,
) -> Dict:
    """
    Replay a fixed first-stage schedule (HVAC on/off and battery mode by hour)
    against an *out-of-sample* test scenario set. The recourse — temperature,
    SoC, battery power, comfort slacks — is recomputed deterministically for
    each test scenario, *without* re-optimizing.

    This is exactly what's needed to report a Robust Feasibility Rate:
       RFR = fraction of test scenarios in which the fixed schedule still
             keeps comfort within tolerance.

    Parameters
    ----------
    first_stage_schedule : output["schedule"] from optimize_schedule_stochastic
    baseline_context : same baseline used at solve time
    test_scenarios : list of Scenario, drawn with a *different* seed than the
                     training scenarios
    mapped : same mapped preferences (used for T_min and guest_window)
    thermal_params, pv_battery_params : see core_stochastic
    comfort_buffer_c : optional tolerance, e.g. 0.5 °C below T_min still counts
                       as feasible

    Returns
    -------
    {
      "rfr": float in [0, 1],
      "per_scenario": [
        {"scenario_id": ..., "feasible": bool, "min_temp_seen": float,
         "comfort_violation_minutes": int, "soc_end_kwh": float, "cost": float},
        ...
      ],
      "cvar_comfort_oos": float,
      "expected_cv_oos": float,
    }
    """
    if thermal_params is None:
        from core_stochastic import Thermal2R2C
        thermal_params = Thermal2R2C()
    if pv_battery_params is None:
        from core_stochastic import PVBattery
        pv_battery_params = PVBattery()

    th = thermal_params
    pvb = pv_battery_params

    h_start = int(baseline_context["horizon_start_hour"])
    h_end = int(baseline_context["horizon_end_hour"])
    hours = list(range(h_start, h_end))
    H = len(hours)

    T_in_0 = float(baseline_context.get("initial_temp_c", 20.0))
    T_m_0 = float(baseline_context.get("initial_mass_temp_c", T_in_0))
    soc_0 = pvb.soc_init_frac * pvb.energy_kwh
    soc_min = pvb.soc_min_frac * pvb.energy_kwh
    soc_max = pvb.soc_max_frac * pvb.energy_kwh

    T_min = float(mapped["min_temp_c"])
    guest_window = set(mapped.get("guest_window", []))

    # Build hour -> schedule-row map
    sched_by_hour = {int(r["hour"]): r for r in first_stage_schedule}

    per_scenario = []
    cv_vals, cv_probs = [], []

    a_ia = 1.0 / th.R_ia
    a_im = 1.0 / th.R_im

    for sc in test_scenarios:
        T_in = T_in_0
        T_m = T_m_0
        soc = soc_0
        cv_min = 0
        min_temp_seen = T_in
        cost = 0.0

        for t, h in enumerate(hours):
            row = sched_by_hour.get(h, {"hvac_on": 0, "battery_mode": "discharge"})
            y_t = int(row["hvac_on"])
            charging = (row.get("battery_mode") == "charge")

            t_out = sc.outdoor_temp_c.get(h, 8.0)
            pv = sc.pv_kw.get(h, 0.0)
            price = sc.price.get(h, 2.0)

            # 2R2C forward Euler, Δt = 1 h
            dT_in = (
                (t_out - T_in) * a_ia / th.C_in
                + (T_m - T_in) * a_im / th.C_in
                + th.kappa_kw * y_t / th.C_in
                + th.Q_int_kw / th.C_in
            )
            dT_m = (T_in - T_m) * a_im / th.C_m
            T_in_new = T_in + dT_in
            T_m_new = T_m + dT_m

            # Battery dispatch under a simple replay rule: try to soak up PV
            # surplus if charging, try to cover HVAC if discharging.
            hvac_kw = th.kappa_kw if y_t else 0.0
            if charging:
                # charge with whatever PV surplus or up to p_max
                surplus = max(0.0, pv - hvac_kw)
                p_ch = min(pvb.p_max_kw, surplus, max(0.0, (soc_max - soc) / pvb.eta_ch))
                p_dis = 0.0
            else:
                p_ch = 0.0
                deficit = max(0.0, hvac_kw - pv)
                p_dis = min(pvb.p_max_kw, deficit, max(0.0, (soc - soc_min) * pvb.eta_dis))

            soc = soc + pvb.eta_ch * p_ch - p_dis / pvb.eta_dis
            net_grid = hvac_kw + p_ch - p_dis - pv  # kW; positive = import
            if net_grid >= 0:
                cost += price * pvb.price_buy_scale * net_grid
            else:
                cost += price * pvb.price_sell_scale * net_grid   # negative

            # Comfort violation tracking on guest window
            if h in guest_window:
                if T_in_new < (T_min - comfort_buffer_c):
                    cv_min += 60   # one hour
                min_temp_seen = min(min_temp_seen, T_in_new)

            T_in, T_m = T_in_new, T_m_new

        feasible = (cv_min == 0)
        per_scenario.append({
            "scenario_id": sc.scenario_id,
            "feasible": feasible,
            "min_temp_seen": round(min_temp_seen, 2),
            "comfort_violation_minutes": cv_min,
            "soc_end_kwh": round(soc, 2),
            "cost": round(cost, 3),
            "probability": sc.probability,
        })
        cv_vals.append(cv_min)
        cv_probs.append(sc.probability)

    n = len(test_scenarios)
    rfr = sum(1 for ps in per_scenario if ps["feasible"]) / n if n else 0.0
    return {
        "rfr": round(rfr, 3),
        "per_scenario": per_scenario,
        "cvar_comfort_oos": round(cvar(cv_vals, cv_probs, alpha=0.2), 2),
        "expected_cv_oos": round(sum(p * v for p, v in zip(cv_probs, cv_vals)), 2),
    }


# --------------------------------------------------------------------------- #
# Pareto frontier sweep helper
# --------------------------------------------------------------------------- #

@dataclass
class ParetoPoint:
    lam_comf: float
    lam_cost: float
    expected_cv: float
    expected_cost: float
    dr_compliance: float
    objective: float


def pareto_sweep(
    solve_fn: Callable[[float, float], Dict],
    lam_comf_grid: Optional[List[float]] = None,
    lam_cost_grid: Optional[List[float]] = None,
) -> List[ParetoPoint]:
    """
    Sweep (lam_comf, lam_cost) on a grid and call `solve_fn(lam_comf, lam_cost)`,
    which is expected to return an `optimize_schedule_stochastic`-style dict.

    Returns a list of ParetoPoint records suitable for plotting a comfort–cost
    Pareto frontier per parser × policy configuration.

    Example
    -------
        from core_stochastic import optimize_schedule_stochastic
        def solve(lc, lp):
            mapped_local = dict(mapped); mapped_local["comfort_weight"] = lc
            mapped_local["cost_weight"] = lp
            return optimize_schedule_stochastic(mapped_local, ctx, scs, ...)
        points = pareto_sweep(solve)
    """
    if lam_comf_grid is None:
        lam_comf_grid = [0.5, 1.0, 2.0, 5.0, 10.0]
    if lam_cost_grid is None:
        lam_cost_grid = [0.5, 1.0, 2.0, 5.0]

    points: List[ParetoPoint] = []
    for lc in lam_comf_grid:
        for lp in lam_cost_grid:
            out = solve_fn(lc, lp)
            if out.get("status") != "ok":
                continue
            mt = out["metrics"]
            points.append(ParetoPoint(
                lam_comf=lc, lam_cost=lp,
                expected_cv=mt.get("expected_comfort_violation_minutes", float("nan")),
                expected_cost=mt.get("expected_cost", float("nan")),
                dr_compliance=mt.get("dr_compliance_score", float("nan")),
                objective=mt.get("objective_value", float("nan")),
            ))
    return points


def is_dominated(p: ParetoPoint, others: List[ParetoPoint]) -> bool:
    """A point is Pareto-dominated if some other point is ≤ on both axes
    and < on at least one."""
    for q in others:
        if q is p:
            continue
        if (q.expected_cv <= p.expected_cv and q.expected_cost <= p.expected_cost
                and (q.expected_cv < p.expected_cv or q.expected_cost < p.expected_cost)):
            return True
    return False


def pareto_front(points: List[ParetoPoint]) -> List[ParetoPoint]:
    """Filter the input to its Pareto-front subset on (expected_cv, expected_cost)."""
    return [p for p in points if not is_dominated(p, points)]
