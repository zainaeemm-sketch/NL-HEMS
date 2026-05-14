"""
src/core_stochastic.py
=======================
Journal-version stochastic CP-SAT solver for NL-HEMS-DR.

This module sits alongside the existing `src/core.py` (which implements the
conference-paper solver) and adds four things the conference version lacks:

  1. A two-state 2R2C thermal model (indoor air + thermal mass) with solar gains
  2. PV and battery storage as first-class decision variables
  3. A scenario-loop wrapper that turns the deterministic CP-SAT model into a
     sample-average approximation (SAA) of the two-stage stochastic problem,
     with first-stage HVAC and battery-mode decisions shared across scenarios
     and recourse variables (temperatures, SoC, comfort slacks) per scenario
  4. Chance-constrained comfort via a Big-M / Σπ_ω z_ω ≤ α reformulation

Public API:
    optimize_schedule_stochastic(mapped, baseline_context, scenarios,
                                  pv_battery_params=None, alpha_comfort=0.0,
                                  enforce_guest_hard_comfort=False)

It returns the same `{"status": ..., "schedule": [...]}` shape as the original
`optimize_schedule`, plus a few extra fields (per-scenario trajectories,
battery SoC, grid import/export) that the dashboard can plot.

Scaling note: CP-SAT operates on integer variables only. We scale temperatures
by 100 (so 0.01 °C resolution) and powers/energies by 1000 (W, Wh). This is the
same trick the conference solver almost certainly uses; if your `core.py`
uses a different scale, align `T_SCALE` and `P_SCALE` below.

Author: NL-HEMS journal extension
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from ortools.sat.python import cp_model


# --------------------------------------------------------------------------- #
# Scaling and defaults
# --------------------------------------------------------------------------- #

T_SCALE = 100        # integer temp units = 0.01 °C
P_SCALE = 1000       # integer power units = 1 W
BIG_M_T = 100 * 100  # 100 °C in scaled units; safe Big-M for comfort indicator


# --------------------------------------------------------------------------- #
# Parameter containers
# --------------------------------------------------------------------------- #

@dataclass
class Thermal2R2C:
    """
    Discrete-time 2R2C parameters. The discretization assumes Δt = 1 hour.

    State update (scaled to integers in the solver):
        T_in(t+1) = T_in(t) + Δt/C_in * [ (T_out - T_in)/R_ia
                                          + (T_m - T_in)/R_im
                                          + κ * y(t)
                                          + Q_int + γ_in_sol * I_sol ]
        T_m (t+1) = T_m (t) + Δt/C_m  * [ (T_in - T_m )/R_im
                                          + γ_m_sol  * I_sol ]
    """
    C_in: float = 1.5       # kWh / K  (air node thermal capacitance)
    C_m: float = 8.0        # kWh / K  (mass node thermal capacitance)
    R_ia: float = 6.0       # K / kW   (indoor air ↔ ambient resistance)
    R_im: float = 2.5       # K / kW   (indoor air ↔ thermal mass resistance)
    kappa_kw: float = 3.0   # kW       (HVAC thermal power, +ve = heating)
    Q_int_kw: float = 0.3   # kW       (internal gains, occupants+appliances)


@dataclass
class PVBattery:
    """PV and battery parameters."""
    pv_enabled: bool = True
    battery_enabled: bool = True

    # Battery sizing
    energy_kwh: float = 10.0
    p_max_kw: float = 5.0
    eta_ch: float = 0.95
    eta_dis: float = 0.95
    soc_init_frac: float = 0.5     # initial SoC as fraction of energy_kwh
    soc_min_frac: float = 0.1
    soc_max_frac: float = 0.95

    # Grid pricing
    price_buy_scale: float = 1.0   # multiplier applied to price[h]
    price_sell_scale: float = 0.5  # feed-in tariff fraction of buy price

    # Other household load (kW) by hour. If None, treated as 0.
    household_load_kw: Optional[Dict[int, float]] = None


# --------------------------------------------------------------------------- #
# Helper to extract per-horizon arrays from a context dict
# --------------------------------------------------------------------------- #

def _ctx_to_arrays(ctx: Dict, hours: List[int]) -> Dict[str, List[float]]:
    """Turn the dict-shaped context into per-hour lists indexed by horizon position."""
    price = ctx.get("price", {})
    t_out = ctx.get("outdoor_temp_c", {})
    pv = ctx.get("pv_kw", {})
    dr_hours = set(ctx.get("dr_event_hours", []))

    return {
        "price": [float(price.get(h, 2.0)) for h in hours],
        "t_out": [float(t_out.get(h, 8.0)) for h in hours],
        "pv": [float(pv.get(h, 0.0)) for h in hours],
        "dr": [1 if h in dr_hours else 0 for h in hours],
    }


# --------------------------------------------------------------------------- #
# Main solver
# --------------------------------------------------------------------------- #

def optimize_schedule_stochastic(
    mapped: Dict,
    baseline_context: Dict,
    scenarios: List,
    pv_battery_params: Optional[PVBattery] = None,
    thermal_params: Optional[Thermal2R2C] = None,
    alpha_comfort: float = 0.0,
    enforce_guest_hard_comfort: bool = False,
    time_limit_s: float = 30.0,
) -> Dict:
    """
    Solve the journal-version two-stage stochastic HEMS via SAA.

    Parameters
    ----------
    mapped : dict
        Output of `map_fuzzy_preferences` (or `map_preferences_type2`).
        Must contain at least:
          - target_temp_c, min_temp_c
          - comfort_weight, cost_weight, dr_weight, switching_weight (≥ 0)
          - guest_window: list of hours [t_s, t_s+1, ..., t_e]
        Optional:
          - guest_event (bool)
    baseline_context : dict
        Initial conditions and parameters shared by all scenarios.
        Required keys: horizon_start_hour, horizon_end_hour, initial_temp_c.
        Optional: initial_mass_temp_c (defaults to initial_temp_c).
    scenarios : list of Scenario
        Output of `generate_scenarios(...)`. Length = N_s.
    pv_battery_params, thermal_params : see classes above.
    alpha_comfort : float in [0,1]
        Chance-constraint level: Pr[T_in ≥ T_min during W] ≥ 1 - alpha_comfort.
        Set to 0 for a hard constraint, or e.g. 0.1 for a 10 %-VaR comfort rule.
    enforce_guest_hard_comfort : bool
        If True AND guest_event, the chance constraint is applied to T_min on
        the guest window W. Otherwise comfort enters only via the soft penalty.
    time_limit_s : float
        CP-SAT wall-clock budget.

    Returns
    -------
    dict with keys:
      status: "ok" | "infeasible" | "timeout"
      schedule:        per-hour first-stage schedule (HVAC, battery mode)
      scenario_traces: per-scenario trajectories (T_in, T_m, SoC, P_grid, slacks)
      metrics:         aggregated metrics (comfort_violation_minutes_expected,
                       cvar_comfort, dr_compliance_score, scr,
                       expected_cost, objective_value, hvac_on_hours)
    """
    if not scenarios:
        return {"status": "infeasible", "error": "no scenarios supplied"}

    pvb = pv_battery_params or PVBattery()
    th = thermal_params or Thermal2R2C()

    # ----------------------------------------------------------------- horizon
    h_start = int(baseline_context["horizon_start_hour"])
    h_end = int(baseline_context["horizon_end_hour"])
    hours = list(range(h_start, h_end))
    H = len(hours)

    # ----------------------------------------------------------------- fuzzy θ
    T_tar = float(mapped["target_temp_c"])
    T_min = float(mapped["min_temp_c"])
    lam_comf = max(0.0, float(mapped.get("comfort_weight", 1.0)))
    lam_cost = max(0.0, float(mapped.get("cost_weight", 1.0)))
    lam_dr = max(0.0, float(mapped.get("dr_weight", 1.0)))
    lam_sw = max(0.0, float(mapped.get("switching_weight", 0.05)))
    lam_min = max(0.0, float(mapped.get("min_comfort_weight", 2.0 * lam_comf)))
    guest_event = bool(mapped.get("guest_event", False))
    guest_window = [int(h) for h in mapped.get("guest_window", [])]

    # Comfort-active indicator ω(t)
    omega = [1 if hours[i] in guest_window else 0 for i in range(H)]

    # ----------------------------------------------------------------- initial
    T_in_0 = int(round(float(baseline_context.get("initial_temp_c", 20.0)) * T_SCALE))
    T_m_0 = int(round(float(baseline_context.get("initial_mass_temp_c",
                                                  baseline_context.get("initial_temp_c", 20.0))) * T_SCALE))
    soc_0 = int(round(pvb.soc_init_frac * pvb.energy_kwh * P_SCALE))
    soc_min = int(round(pvb.soc_min_frac * pvb.energy_kwh * P_SCALE))
    soc_max = int(round(pvb.soc_max_frac * pvb.energy_kwh * P_SCALE))
    soc_cap = int(round(pvb.energy_kwh * P_SCALE))

    # ----------------------------------------------------------------- model
    m = cp_model.CpModel()

    # --- first-stage decisions (shared across scenarios) ---
    y = [m.NewBoolVar(f"y_{t}") for t in range(H)]              # HVAC on/off
    u_bat = [m.NewBoolVar(f"ubat_{t}") for t in range(H)]       # 1 = charging

    # switching variables (also first-stage)
    s = [m.NewIntVar(0, 1, f"s_{t}") for t in range(H)]
    for t in range(H):
        if t == 0:
            m.Add(s[t] >= y[t])         # vs y(-1) = 0
            m.Add(s[t] >= -y[t])
        else:
            m.Add(s[t] >= y[t] - y[t - 1])
            m.Add(s[t] >= y[t - 1] - y[t])

    # --- precompute scaled 2R2C coefficients (constant across scenarios) ---
    # Using forward-Euler with Δt = 1 h.
    a_ia = 1.0 / th.R_ia
    a_im = 1.0 / th.R_im
    coef_T_in_self = 1.0 - (a_ia + a_im) / th.C_in
    coef_T_in_out = a_ia / th.C_in
    coef_T_in_m = a_im / th.C_in
    coef_T_in_y = th.kappa_kw / th.C_in
    coef_T_in_int = th.Q_int_kw / th.C_in
    coef_T_m_self = 1.0 - a_im / th.C_m
    coef_T_m_in = a_im / th.C_m

    def _to_int_coef(x: float) -> int:
        return int(round(x * 1000))     # 0.001 resolution on the coefficient

    # We scale the dynamics equation by 1000 to keep CP-SAT in integers.

    z_violations: List[cp_model.IntVar] = []          # chance-constraint flags
    obj_terms: List[Tuple[int, cp_model.IntVar]] = [] # (coef_int, var)

    scenario_vars: List[Dict] = []   # store handles for later extraction

    # =========================== per-scenario block ========================== #
    for sc in scenarios:
        sc_id = sc.scenario_id
        pi = sc.probability
        arrs = _ctx_to_arrays(sc.as_context_override(baseline_context), hours)
        prices = arrs["price"]
        t_outs = arrs["t_out"]
        pv_kws = arrs["pv"]
        dr_flags = arrs["dr"]

        # --- recourse state variables ---
        T_in = [m.NewIntVar(-50 * T_SCALE, 60 * T_SCALE, f"Tin_s{sc_id}_t{t}") for t in range(H + 1)]
        T_m = [m.NewIntVar(-50 * T_SCALE, 60 * T_SCALE, f"Tm_s{sc_id}_t{t}") for t in range(H + 1)]
        m.Add(T_in[0] == T_in_0)
        m.Add(T_m[0] == T_m_0)

        # --- battery recourse (per scenario) ---
        P_ch = [m.NewIntVar(0, int(pvb.p_max_kw * P_SCALE), f"Pch_s{sc_id}_t{t}") for t in range(H)]
        P_dis = [m.NewIntVar(0, int(pvb.p_max_kw * P_SCALE), f"Pdis_s{sc_id}_t{t}") for t in range(H)]
        SoC = [m.NewIntVar(0, soc_cap, f"SoC_s{sc_id}_t{t}") for t in range(H + 1)]
        m.Add(SoC[0] == soc_0)
        # Mutual exclusion: charge only if u_bat=1, discharge only if u_bat=0
        big_P = int(pvb.p_max_kw * P_SCALE)
        for t in range(H):
            m.Add(P_ch[t] <= big_P * u_bat[t])
            m.Add(P_dis[t] <= big_P * (1 - u_bat[t]))
            m.Add(SoC[t] >= soc_min)
            m.Add(SoC[t] <= soc_max)

            # SoC dynamics (integer-scaled, with rounding tolerance):
            #   SoC[t+1] = SoC[t] + eta_ch * P_ch - P_dis / eta_dis  (Δt = 1 h)
            # We scale by 1000 to absorb the fractional efficiencies, then allow
            # ±500 (= ±0.5 Wh) rounding slack so the integer solver can satisfy it.
            eta_ch_int = int(round(pvb.eta_ch * 1000))
            eta_dis_int = int(round(1000 / pvb.eta_dis))
            soc_rhs = 1000 * SoC[t] + eta_ch_int * P_ch[t] - eta_dis_int * P_dis[t]
            m.Add(1000 * SoC[t + 1] <= soc_rhs + 500)
            m.Add(1000 * SoC[t + 1] >= soc_rhs - 500)

        # --- grid import/export (per scenario) ---
        P_grid_pos = [m.NewIntVar(0, 20 * P_SCALE, f"Gp_s{sc_id}_t{t}") for t in range(H)]
        P_grid_neg = [m.NewIntVar(0, 20 * P_SCALE, f"Gn_s{sc_id}_t{t}") for t in range(H)]
        for t in range(H):
            load_kw = 0.0
            if pvb.household_load_kw:
                load_kw = float(pvb.household_load_kw.get(hours[t], 0.0))
            # Power balance:  Gp - Gn = P_HVAC*y + P_load + P_ch - P_dis - P_PV
            rhs_const_int = int(round((load_kw - pv_kws[t]) * P_SCALE))
            p_hvac_int = int(round(th.kappa_kw * P_SCALE))   # approx using HVAC thermal kW
            m.Add(P_grid_pos[t] - P_grid_neg[t]
                  == p_hvac_int * y[t] + rhs_const_int + P_ch[t] - P_dis[t])

        # --- 2R2C thermal dynamics (per scenario) ---
        # Scale equation by 1000:
        # 1000 * T_in[t+1] = c_self * T_in[t] + c_out * T_out + c_m * T_m + c_y * y + c_int (constants in integer)
        c_in_self = _to_int_coef(coef_T_in_self)
        c_in_out = _to_int_coef(coef_T_in_out)
        c_in_m = _to_int_coef(coef_T_in_m)
        c_in_y = _to_int_coef(coef_T_in_y) * T_SCALE   # y is 0/1, want °C·100
        c_in_int = _to_int_coef(coef_T_in_int) * T_SCALE
        c_m_self = _to_int_coef(coef_T_m_self)
        c_m_in = _to_int_coef(coef_T_m_in)

        for t in range(H):
            t_out_scaled = int(round(t_outs[t] * T_SCALE))
            # T_in dynamics with ±500 (= ±0.005 °C) rounding slack on the
            # scaled-by-1000 equation. This is required because the RHS is an
            # integer linear expression that is *not generally divisible by
            # 1000* — so an exact == constraint is infeasible by construction.
            t_in_rhs = (c_in_self * T_in[t] + c_in_out * t_out_scaled
                        + c_in_m * T_m[t] + c_in_y * y[t] + c_in_int)
            m.Add(1000 * T_in[t + 1] <= t_in_rhs + 500)
            m.Add(1000 * T_in[t + 1] >= t_in_rhs - 500)

            # T_m dynamics with the same tolerance band.
            t_m_rhs = c_m_self * T_m[t] + c_m_in * T_in[t]
            m.Add(1000 * T_m[t + 1] <= t_m_rhs + 500)
            m.Add(1000 * T_m[t + 1] >= t_m_rhs - 500)

        # --- comfort slacks (per scenario) ---
        T_tar_int = int(round(T_tar * T_SCALE))
        T_min_int = int(round(T_min * T_SCALE))
        e = [m.NewIntVar(0, 100 * T_SCALE, f"e_s{sc_id}_t{t}") for t in range(H)]
        h_slack = [m.NewIntVar(0, 100 * T_SCALE, f"h_s{sc_id}_t{t}") for t in range(H)]
        for t in range(H):
            m.Add(e[t] >= T_tar_int - T_in[t + 1])
            m.Add(h_slack[t] >= T_min_int - T_in[t + 1])

        # --- chance constraint on guest comfort ---
        if enforce_guest_hard_comfort and guest_event and any(omega):
            if alpha_comfort <= 0.0:
                # hard constraint, per scenario
                for t in range(H):
                    if omega[t]:
                        m.Add(T_in[t + 1] >= T_min_int)
            else:
                # one violation flag per scenario covers any t in W
                z_sc = m.NewBoolVar(f"z_violate_s{sc_id}")
                for t in range(H):
                    if omega[t]:
                        m.Add(T_in[t + 1] >= T_min_int - BIG_M_T * z_sc)
                z_violations.append((pi, z_sc))

        # --- objective contribution (scaled, weighted by pi) ---
        # Note: CP-SAT minimizes integer linear; we accumulate (coef_int, var) and
        # build the objective at the end.
        pi_int = int(round(pi * 1000))         # probability scaled by 1000

        # cost: pi * lam_cost * (price_buy * Gp - price_sell * Gn) per hour
        for t in range(H):
            buy = pvb.price_buy_scale * prices[t]
            sell = pvb.price_sell_scale * prices[t]
            obj_terms.append((pi_int * int(round(lam_cost * buy)), P_grid_pos[t]))
            obj_terms.append((-pi_int * int(round(lam_cost * sell)), P_grid_neg[t]))
            # DR penalty: pi * lam_dr * d(t) * P_HVAC * y
            if dr_flags[t]:
                obj_terms.append(
                    (pi_int * int(round(lam_dr * th.kappa_kw * P_SCALE)), y[t])
                )
            # Comfort penalties (only on active hours)
            if omega[t]:
                obj_terms.append((pi_int * int(round(lam_comf)), e[t]))
                obj_terms.append((pi_int * int(round(lam_min)), h_slack[t]))

        scenario_vars.append({
            "scenario": sc, "T_in": T_in, "T_m": T_m, "SoC": SoC,
            "P_ch": P_ch, "P_dis": P_dis, "P_grid_pos": P_grid_pos,
            "P_grid_neg": P_grid_neg, "e": e, "h_slack": h_slack,
        })

    # --- switching cost (first-stage, no scenario weighting) ---
    sw_coef = int(round(lam_sw * 1000))
    for t in range(H):
        obj_terms.append((sw_coef, s[t]))

    # --- chance-constraint aggregate ---
    if z_violations:
        # Σ pi * z ≤ alpha
        m.Add(sum(int(round(pi_v * 1000)) * z_v for pi_v, z_v in z_violations)
              <= int(round(alpha_comfort * 1000)))

    # --- assemble objective ---
    m.Minimize(sum(c * v for c, v in obj_terms))

    # --- solve ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    status = solver.Solve(m)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Provide actionable diagnostics
        status_name = {
            cp_model.UNKNOWN: "UNKNOWN",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
            cp_model.INFEASIBLE: "INFEASIBLE",
        }.get(status, f"status_code_{int(status)}")
        diag = {
            "solver_status_name": status_name,
            "n_scenarios": len(scenarios),
            "alpha_comfort": alpha_comfort,
            "guest_event": guest_event,
            "guest_window": guest_window,
            "T_min_c": T_min,
            "T_tar_c": T_tar,
            "initial_T_in_c": T_in_0 / T_SCALE,
            "horizon_hours": [hours[0], hours[-1]],
        }
        hint = []
        if enforce_guest_hard_comfort and guest_event and any(omega):
            hint.append(
                "Hard guest comfort is active. Try: set alpha_comfort higher "
                "(e.g. 0.3), lower mapped['min_temp_c'] by 0.5-1.0 °C, or set "
                "enforce_guest_hard_comfort=False to test soft mode first."
            )
        if T_min - (T_in_0 / T_SCALE) > 1.5:
            hint.append(
                f"T_min ({T_min}) is {T_min - T_in_0/T_SCALE:.1f} °C above the initial "
                "indoor temp. The HVAC may not be able to warm the room in time before "
                "the guest window starts. Try a lower T_min, an earlier horizon start, "
                "or a larger Thermal2R2C(kappa_kw=...) to model a stronger heater."
            )
        return {
            "status": "infeasible",
            "solver_status": int(status),
            "diagnostics": diag,
            "hints": hint,
            "schedule": [], "scenario_traces": [], "metrics": {},
        }

    # --- extract first-stage schedule ---
    schedule_rows = []
    for t, h in enumerate(hours):
        schedule_rows.append({
            "hour": h,
            "hvac_on": int(solver.Value(y[t])),
            "battery_mode": "charge" if solver.Value(u_bat[t]) else "discharge",
            "switching": int(solver.Value(s[t])),
            "dr_event": 1 if h in (baseline_context.get("dr_event_hours") or []) else 0,
            "guest_window": int(omega[t]),
        })

    # --- extract per-scenario trajectories + metrics ---
    traces = []
    cv_per_scenario = []
    dr_compliance_acc = 0.0
    dr_hours_acc = 0
    scr_num = 0.0
    scr_den = 0.0
    cost_acc = 0.0
    hvac_on_total = sum(int(solver.Value(y[t])) for t in range(H))
    avg_temp_acc = 0.0

    for sv in scenario_vars:
        sc = sv["scenario"]
        rows = []
        cv = 0
        pv_avail = 0.0
        pv_used = 0.0
        load_total = 0.0
        arrs = _ctx_to_arrays(sc.as_context_override(baseline_context), hours)
        cost_s = 0.0
        for t, h in enumerate(hours):
            tin = solver.Value(sv["T_in"][t + 1]) / T_SCALE
            tm = solver.Value(sv["T_m"][t + 1]) / T_SCALE
            soc = solver.Value(sv["SoC"][t + 1]) / P_SCALE
            pch = solver.Value(sv["P_ch"][t]) / P_SCALE
            pdis = solver.Value(sv["P_dis"][t]) / P_SCALE
            gpos = solver.Value(sv["P_grid_pos"][t]) / P_SCALE
            gneg = solver.Value(sv["P_grid_neg"][t]) / P_SCALE

            rows.append({
                "hour": h, "T_in": round(tin, 2), "T_m": round(tm, 2),
                "SoC_kwh": round(soc, 2), "P_ch_kw": round(pch, 2),
                "P_dis_kw": round(pdis, 2), "P_grid_kw": round(gpos - gneg, 2),
                "price": round(arrs["price"][t], 3),
                "T_out": round(arrs["t_out"][t], 2),
                "PV_kw": round(arrs["pv"][t], 2),
            })

            if omega[t] and tin < (T_min - 1e-6):
                cv += 1

            cost_s += arrs["price"][t] * (pvb.price_buy_scale * gpos
                                          - pvb.price_sell_scale * gneg)

            # DR compliance: HVAC off during DR
            if arrs["dr"][t]:
                dr_hours_acc += 1
                if int(solver.Value(y[t])) == 0:
                    dr_compliance_acc += 1

            # SCR numerator/denominator
            pv_kw = arrs["pv"][t]
            pv_avail += pv_kw
            local_demand = (th.kappa_kw if int(solver.Value(y[t])) == 1 else 0.0) + pch
            pv_used += min(pv_kw, local_demand)

            avg_temp_acc += tin

        cv_minutes = cv * 60   # one violation per hour → 60 minutes
        cv_per_scenario.append((sc.probability, cv_minutes))
        cost_acc += sc.probability * cost_s
        if pv_avail > 1e-9:
            scr_num += sc.probability * pv_used
            scr_den += sc.probability * pv_avail

        traces.append({
            "scenario_id": sc.scenario_id,
            "probability": sc.probability,
            "rows": rows,
            "comfort_violation_minutes": cv_minutes,
            "scenario_cost": round(cost_s, 3),
        })

    # CVaR_alpha on comfort violation (α = 0.2 by default for reporting)
    alpha_cvar = 0.2
    cv_sorted = sorted(cv_per_scenario, key=lambda x: -x[1])  # worst first
    cum_p = 0.0
    cvar_num = 0.0
    for p, cv in cv_sorted:
        if cum_p >= alpha_cvar:
            break
        take = min(p, alpha_cvar - cum_p)
        cvar_num += take * cv
        cum_p += take
    cvar = cvar_num / alpha_cvar if alpha_cvar > 0 else 0.0

    expected_cv = sum(p * cv for p, cv in cv_per_scenario)
    metrics = {
        "expected_comfort_violation_minutes": round(expected_cv, 2),
        "cvar_comfort_violation_minutes": round(cvar, 2),
        "dr_compliance_score": (dr_compliance_acc / dr_hours_acc) if dr_hours_acc else 1.0,
        "self_consumption_ratio": (scr_num / scr_den) if scr_den > 1e-9 else None,
        "expected_cost": round(cost_acc, 3),
        "hvac_on_hours": hvac_on_total,
        "avg_temp_across_scenarios": round(avg_temp_acc / (H * len(scenarios)), 2),
        "objective_value": round(solver.ObjectiveValue() / 1000.0, 3),
        "n_scenarios": len(scenarios),
    }

    return {
        "status": "ok",
        "solver_status": int(status),
        "schedule": schedule_rows,
        "scenario_traces": traces,
        "metrics": metrics,
    }


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from scenarios import generate_scenarios, ForecastErrorModel

    baseline_ctx = {
        "horizon_start_hour": 16, "horizon_end_hour": 24,
        "initial_temp_c": 19.5,
        "price": {16: 2, 17: 3, 18: 8, 19: 9, 20: 8, 21: 5, 22: 3, 23: 2},
        "dr_event_hours": [19, 20],
    }
    mapped = {
        "target_temp_c": 21.5, "min_temp_c": 20.5,
        "comfort_weight": 5.0, "cost_weight": 1.0, "dr_weight": 3.0,
        "switching_weight": 0.1, "min_comfort_weight": 8.0,
        "guest_event": True,
        "guest_window": [18, 19, 20, 21, 22],
    }
    scs = generate_scenarios(baseline_ctx, n_scenarios=5,
                              error_model=ForecastErrorModel(seed=0))

    # Start with soft comfort (no chance constraint, no hard guest enforcement)
    # so the very first run always succeeds. Then re-run with hard mode + a
    # generous chance level so a violation is *allowed* in 1 of 5 scenarios.
    print("=" * 60)
    print("Run 1: soft comfort (always feasible by construction)")
    print("=" * 60)
    out1 = optimize_schedule_stochastic(
        mapped, baseline_ctx, scs,
        pv_battery_params=PVBattery(),
        thermal_params=Thermal2R2C(),
        alpha_comfort=0.0,
        enforce_guest_hard_comfort=False,
    )
    print("status:", out1["status"])
    print("metrics:", out1.get("metrics"))
    if out1.get("status") == "ok":
        print("first-stage schedule:")
        for r in out1["schedule"]:
            print(" ", r)

    print()
    print("=" * 60)
    print("Run 2: hard guest comfort with α=0.2 (1-of-5 scenarios may violate)")
    print("=" * 60)
    out2 = optimize_schedule_stochastic(
        mapped, baseline_ctx, scs,
        pv_battery_params=PVBattery(),
        thermal_params=Thermal2R2C(),
        alpha_comfort=0.2,
        enforce_guest_hard_comfort=True,
    )
    print("status:", out2["status"])
    if out2.get("status") == "ok":
        print("metrics:", out2["metrics"])
    else:
        print("diagnostics:", out2.get("diagnostics"))
        print("hints:")
        for h in out2.get("hints", []):
            print(" -", h)
