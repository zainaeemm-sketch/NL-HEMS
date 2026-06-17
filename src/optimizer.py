"""
Two-stage stochastic mixed-integer optimizer in OR-Tools CP-SAT.

Implements the formulation of Sections V-A through V-D of the paper:
  - 2R2C thermal envelope
  - PV / battery dynamics
  - Two-stage stochastic structure with HVAC + battery mode as
    first-stage decisions
  - Big-M chance-constrained comfort (Eq. 21-23)

All variables are integers in scaled units:
  Temperature : x100   (0.01 C resolution)
  Power       : x1000  (1 W resolution)
  Energy/SoC  : x1000  (1 Wh resolution)
  Cost        : x1000

Returns a dict with the schedule and per-scenario trajectories.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List
from ortools.sat.python import cp_model

# Scaling factors
SC_T = 100      # temperature
SC_P = 1000     # power (W)
SC_E = 1000     # energy (Wh)
SC_C = 1000     # cost / objective

# Default building / asset parameters (paper Section VII-A)
DEFAULT_PARAMS: Dict[str, Any] = {
    "C_in": 1.5,         # kWh/K
    "C_m":  8.0,         # kWh/K
    "R_ia": 6.0,         # K/kW
    "R_im": 2.5,         # K/kW
    "kappa": 3.0,        # kW HVAC thermal power
    "P_HVAC_elec": 1.5,  # kW HVAC electrical draw
    "Q_int": 0.2,        # kW internal gains
    "P_load_base": 0.4,  # kW non-HVAC base load
    "E_bat": 10.0,       # kWh
    "P_bat": 5.0,        # kW
    "eta_ch": 0.95,
    "eta_dis": 0.95,
    "SoC_min_frac": 0.10,
    "SoC_max_frac": 0.95,
    "SoC_init_frac": 0.50,
    "T_in_init": 21.0,
    "T_m_init":  21.0,
    "p_sell_fraction": 0.45,
    "dt_hours": 1.0,
}


def _merge_params(theta: dict, override: dict | None) -> dict:
    p = dict(DEFAULT_PARAMS)
    if override:
        p.update(override)
    p.update(theta)
    return p


def solve_stochastic(theta: dict,
                     scenarios: List[Dict[str, np.ndarray]],
                     alpha: float = 0.20,
                     guest_window: tuple[int, int] | None = None,
                     building: dict | None = None,
                     time_limit_s: float = 120.0,
                     verbose: bool = False,
                     fix_y: np.ndarray | None = None,
                     fix_ubat: np.ndarray | None = None,
                     hint_y: np.ndarray | None = None,
                     hint_ubat: np.ndarray | None = None) -> Dict[str, Any]:
    """
    Solve the two-stage SAA problem.

    Parameters
    ----------
    theta : dict
        Output of fuzzy / direct mapping.
    scenarios : list of dicts with keys T_out, price, PV, load, d, prob.
    alpha : chance-constraint level. alpha == 0 -> hard constraint on
            indoor temperature inside guest_window.
    guest_window : (start, end) hour range on which the chance constraint
            is active. If None, the constraint is inactive.
    fix_y, fix_ubat : optional first-stage decision sequences to fix
            before solving the recourse. Used to evaluate a previously
            obtained schedule on a new scenario set (e.g. for VSS).
    """
    p = _merge_params(theta, building)
    H = int(len(scenarios[0]["T_out"]))
    N = len(scenarios)
    dt = p["dt_hours"]

    model = cp_model.CpModel()

    # -----------------------------------------------------------------
    # First-stage decisions
    # -----------------------------------------------------------------
    y     = [model.NewBoolVar(f"y_{t}")     for t in range(H)]
    ubat  = [model.NewBoolVar(f"ubat_{t}")  for t in range(H)]
    sw    = [model.NewIntVar(0, 1, f"sw_{t}") for t in range(H)]

    for t in range(1, H):
        model.Add(sw[t] >= y[t]   - y[t-1])
        model.Add(sw[t] >= y[t-1] - y[t])

    # If a previously obtained first-stage schedule is supplied, fix
    # y(t) and u_bat(t) so the optimizer only computes the recourse.
    if fix_y is not None:
        fix_y_arr = np.asarray(fix_y, dtype=int).ravel()
        for t in range(min(H, len(fix_y_arr))):
            model.Add(y[t] == int(fix_y_arr[t]))
    if fix_ubat is not None:
        fix_u_arr = np.asarray(fix_ubat, dtype=int).ravel()
        for t in range(min(H, len(fix_u_arr))):
            model.Add(ubat[t] == int(fix_u_arr[t]))

    # Warm-start: bias the search toward a known feasible plan (e.g. the
    # deterministic solution). The optimum can only improve on it, so the
    # stochastic objective never comes out worse than the warm-start plan.
    if hint_y is not None:
        hy = np.asarray(hint_y, dtype=int).ravel()
        for t in range(min(H, len(hy))):
            model.AddHint(y[t], int(hy[t]))
    if hint_ubat is not None:
        hu = np.asarray(hint_ubat, dtype=int).ravel()
        for t in range(min(H, len(hu))):
            model.AddHint(ubat[t], int(hu[t]))

    # -----------------------------------------------------------------
    # Recourse decisions (per scenario)
    # -----------------------------------------------------------------
    T_in_v = [[None]*(H+1) for _ in range(N)]
    T_m_v  = [[None]*(H+1) for _ in range(N)]
    SoC_v  = [[None]*(H+1) for _ in range(N)]
    Pch_v  = [[None]*H     for _ in range(N)]
    Pdis_v = [[None]*H     for _ in range(N)]
    Pgp_v  = [[None]*H     for _ in range(N)]
    Pgn_v  = [[None]*H     for _ in range(N)]
    e_v    = [[None]*H     for _ in range(N)]
    h_v    = [[None]*H     for _ in range(N)]
    z_v    = [model.NewBoolVar(f"z_{w}") for w in range(N)]

    # Bounds in scaled integer units
    T_BOUND     = (int( 0 * SC_T), int(35 * SC_T))
    SoC_min_int = int(p["SoC_min_frac"] * p["E_bat"] * SC_E)
    SoC_max_int = int(p["SoC_max_frac"] * p["E_bat"] * SC_E)
    SoC_init_int = int(p["SoC_init_frac"] * p["E_bat"] * SC_E)
    P_bat_int   = int(p["P_bat"] * SC_P)
    P_max_grid  = int(20 * SC_P)
    P_HVAC_int  = int(p["P_HVAC_elec"] * SC_P)
    SLACK_MAX   = int(35 * SC_T)   # widened to full T range so slacks
                                   # can absorb any deviation without forcing
                                   # infeasibility
    T_min_int   = int(round(p["T_min"]    * SC_T))
    T_tar_int   = int(round(p["T_target"] * SC_T))

    for w in range(N):
        scen = scenarios[w]
        for t in range(H+1):
            T_in_v[w][t] = model.NewIntVar(*T_BOUND, f"Tin_{w}_{t}")
            T_m_v[w][t]  = model.NewIntVar(*T_BOUND, f"Tm_{w}_{t}")
            SoC_v[w][t]  = model.NewIntVar(SoC_min_int, SoC_max_int,
                                           f"SoC_{w}_{t}")
        model.Add(T_in_v[w][0] == int(p["T_in_init"] * SC_T))
        model.Add(T_m_v[w][0]  == int(p["T_m_init"]  * SC_T))
        model.Add(SoC_v[w][0]  == SoC_init_int)

        for t in range(H):
            Pch_v[w][t]  = model.NewIntVar(0, P_bat_int, f"Pch_{w}_{t}")
            Pdis_v[w][t] = model.NewIntVar(0, P_bat_int, f"Pdis_{w}_{t}")
            Pgp_v[w][t]  = model.NewIntVar(0, P_max_grid, f"Pgp_{w}_{t}")
            Pgn_v[w][t]  = model.NewIntVar(0, P_max_grid, f"Pgn_{w}_{t}")
            e_v[w][t]    = model.NewIntVar(0, SLACK_MAX,  f"e_{w}_{t}")
            h_v[w][t]    = model.NewIntVar(0, SLACK_MAX,  f"h_{w}_{t}")

            # Battery mutual exclusion via first-stage mode
            model.Add(Pch_v[w][t]  <= P_bat_int * ubat[t])
            model.Add(Pdis_v[w][t] <= P_bat_int * (1 - ubat[t]))

            # --- 2R2C thermal dynamics ---------------------------------
            # T_in[t+1] = T_in[t] + dt/Cin * ((T_out - T_in)/R_ia +
            #             (T_m - T_in)/R_im + kappa*y + Q_int)
            # Multiply through by Cin*R_ia*R_im to keep integer arithmetic
            T_out_int = int(round(scen["T_out"][t] * SC_T))
            a1 = dt * p["R_im"]
            a2 = dt * p["R_ia"]
            a3 = dt * p["R_ia"] * p["R_im"] * p["kappa"] * SC_T   # for y*kappa term
            a4 = dt * p["R_ia"] * p["R_im"] * p["Q_int"] * SC_T
            denom = p["C_in"] * p["R_ia"] * p["R_im"]

            # Convert to integer coefficients (with a chosen scaling)
            COEF = 1000
            c1 = int(round(a1 * COEF))
            c2 = int(round(a2 * COEF))
            c3 = int(round(a3 * COEF))
            c4 = int(round(a4 * COEF))
            cD = int(round(denom * COEF))

            lhs = cD * (T_in_v[w][t+1] - T_in_v[w][t])
            rhs = (c1 * (T_out_int - T_in_v[w][t])
                 + c2 * (T_m_v[w][t] - T_in_v[w][t])
                 + c3 * y[t]
                 + c4)
            # Tight inequality band: half-unit rounding tolerance so the
            # integer scaled temperature can take its nearest-integer
            # value to the true real-valued update (paper Section V-E).
            tol = max(1, cD // 2)
            model.Add(lhs - rhs <=  tol)
            model.Add(lhs - rhs >= -tol)

            # T_m dynamics
            b1 = dt * p["R_im"] * SC_T  # but cancels with denominator below
            cD_m = int(round(p["C_m"] * p["R_im"] * COEF))
            c1_m = int(round(dt * COEF))
            lhs_m = cD_m * (T_m_v[w][t+1] - T_m_v[w][t])
            rhs_m = c1_m * (T_in_v[w][t] - T_m_v[w][t])
            tol_m = max(1, cD_m // 2)
            model.Add(lhs_m - rhs_m <=  tol_m)
            model.Add(lhs_m - rhs_m >= -tol_m)

            # --- Battery SoC dynamics --------------------------------
            # SoC[t+1] = SoC[t] + dt * (eta_ch * Pch - Pdis / eta_dis)
            # All in Wh after scaling.
            COEF_E = 1000
            c_ch  = int(round(p["eta_ch"] * dt * COEF_E))
            c_dis = int(round(dt / p["eta_dis"] * COEF_E))
            lhs_e = COEF_E * (SoC_v[w][t+1] - SoC_v[w][t])
            rhs_e = c_ch * Pch_v[w][t] - c_dis * Pdis_v[w][t]
            tol_e = max(1, COEF_E // 2)
            model.Add(lhs_e - rhs_e <=  tol_e)
            model.Add(lhs_e - rhs_e >= -tol_e)

            # --- Power balance ----------------------------------------
            PV_int = int(round(scen["PV"][t] * SC_P))
            # Time-varying base load (kW) from the scenario; fall back
            # to the constant default if the scenario does not carry it
            load_kw = scen["load"][t] if "load" in scen else p["P_load_base"]
            P_load_int_t = int(round(float(load_kw) * SC_P))
            # Pgp - Pgn = P_HVAC*y + P_load(t) + Pch - Pdis - PV
            model.Add(Pgp_v[w][t] - Pgn_v[w][t]
                      == P_HVAC_int * y[t] + P_load_int_t
                       + Pch_v[w][t] - Pdis_v[w][t] - PV_int)

            # --- Comfort slacks ---------------------------------------
            # e >= T_target - T_in    ;  h >= T_min - T_in
            model.Add(e_v[w][t] >= T_tar_int - T_in_v[w][t+1])
            model.Add(h_v[w][t] >= T_min_int - T_in_v[w][t+1])

        # --- Chance constraint ---------------------------------------
        if alpha < 1.0 and guest_window is not None:
            ws, we = guest_window
            ws = max(0, ws); we = min(H, we)
            BIG_M = int(20 * SC_T)
            for t in range(ws, we):
                model.Add(T_in_v[w][t+1] >= T_min_int - BIG_M * z_v[w])

    # Probability-weighted relaxation cap (scenarios are uniform)
    if alpha < 1.0 and guest_window is not None and alpha < 1.0 - 1e-9:
        # sum prob_w * z_w <= alpha -> sum z_w <= alpha*N  (uniform)
        max_relax = int(np.floor(alpha * N))
        model.Add(sum(z_v) <= max_relax)

    # -----------------------------------------------------------------
    # Objective: expected per-scenario cost (Eq. 18-19)
    # -----------------------------------------------------------------
    # Per scenario:
    #   sum_t [ lambda_cost * p_w(t) * (Pgp - p_sell * Pgn) * dt
    #         + lambda_dr   * d(t)   * P_HVAC * y
    #         + lambda_comf * e
    #         + lambda_min  * h
    #         + lambda_sw   * sw ]
    obj_terms = []
    psell = p["p_sell_fraction"]
    lam_cost = float(p["lambda_cost"])
    lam_dr   = float(p["lambda_dr"])
    lam_comf = float(p["lambda_comf"])
    lam_min  = float(p["lambda_min"])
    lam_sw   = float(p["lambda_sw"])

    for w in range(N):
        prob_int = int(round(scenarios[w]["prob"] * 1000))
        for t in range(H):
            # net import - cost
            price_int = int(round(scenarios[w]["price"][t] * 100))   # cents
            buy_term  = int(round(lam_cost * price_int)) * Pgp_v[w][t]
            sell_term = int(round(lam_cost * psell * price_int)) * Pgn_v[w][t]
            dr_term   = int(round(lam_dr * scenarios[w]["d"][t] * P_HVAC_int)) * y[t]
            comf_term = int(round(lam_comf * SC_C)) * e_v[w][t]
            min_term  = int(round(lam_min  * SC_C)) * h_v[w][t]
            sw_term   = int(round(lam_sw   * SC_C * SC_T)) * sw[t]

            # Scale by scenario probability
            obj_terms.append(prob_int * (buy_term - sell_term + dr_term
                                         + comf_term + min_term + sw_term))

    model.Minimize(sum(obj_terms))

    # -----------------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 4
    status = solver.Solve(model)

    out = {
        "status":          solver.StatusName(status),
        "feasible":        status in (cp_model.OPTIMAL, cp_model.FEASIBLE),
        "objective":       solver.ObjectiveValue() if status in
                            (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        "wall_time_s":     solver.WallTime(),
        "best_bound":      (solver.BestObjectiveBound() if status in
                            (cp_model.OPTIMAL, cp_model.FEASIBLE) else None),
        "optimal":         status == cp_model.OPTIMAL,
        "alpha":           alpha,
        "N_scenarios":     N,
        "horizon":         H,
    }

    if out["feasible"]:
        out["y"]    = np.array([solver.Value(y[t]) for t in range(H)])
        out["ubat"] = np.array([solver.Value(ubat[t]) for t in range(H)])
        out["sw"]   = np.array([solver.Value(sw[t])   for t in range(H)])
        out["T_in"] = np.array([[solver.Value(T_in_v[w][t]) / SC_T
                                 for t in range(H+1)] for w in range(N)])
        out["SoC"]  = np.array([[solver.Value(SoC_v[w][t]) / SC_E
                                 for t in range(H+1)] for w in range(N)])
        out["Pgrid_pos"] = np.array([[solver.Value(Pgp_v[w][t]) / SC_P
                                      for t in range(H)] for w in range(N)])
        out["Pgrid_neg"] = np.array([[solver.Value(Pgn_v[w][t]) / SC_P
                                      for t in range(H)] for w in range(N)])
        out["Pch"]       = np.array([[solver.Value(Pch_v[w][t]) / SC_P
                                      for t in range(H)] for w in range(N)])
        out["Pdis"]      = np.array([[solver.Value(Pdis_v[w][t]) / SC_P
                                      for t in range(H)] for w in range(N)])
        out["z_relax"]   = np.array([solver.Value(z_v[w]) for w in range(N)])

    return out
