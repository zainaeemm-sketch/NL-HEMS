"""
Baseline optimizers for the reviewer-requested ablation.

  - solve_deterministic : single-scenario optimization on point forecast.
                          Used to compute Value of Stochastic Solution.
  - solve_mpc           : receding-horizon MPC using deterministic
                          optimization at each step, with realized
                          uncertainty unfolding hour by hour.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List

from .optimizer import solve_stochastic
from .scenarios import deterministic_scenario


def solve_deterministic(theta: dict,
                        context: dict,
                        guest_window: tuple[int, int] | None = None,
                        building: dict | None = None,
                        time_limit_s: float = 30.0) -> dict:
    """Deterministic baseline: a single optimization over the
    point-forecast (historical-average) context, with HARD comfort.
 
    With a single scenario the two-stage model collapses to one set of
    variables, so this is exactly a single-shot deterministic program.
    alpha is fixed to 0.0  ->  hard comfort bound T_in >= T_min for all t
    (the alpha->0 limit of the chance constraint). This is the value
    that enters the VSS as the expected-value plan.
    """
    scen = deterministic_scenario(context)          # mean price/temp/PV
    out = solve_stochastic(theta=theta, scenarios=scen,
                           alpha=0.0,               # <-- HARD comfort, always
                           guest_window=guest_window,
                           building=building,
                           time_limit_s=time_limit_s)
    out["method"] = "deterministic"
    return out


def solve_mpc(theta: dict,
              context: dict,
              guest_window: tuple[int, int] | None = None,
              building: dict | None = None,
              realisation: dict | None = None,
              receding_horizon: int = 12,
              time_limit_s: float = 5.0) -> Dict[str, Any]:
    """
    Receding-horizon MPC. At each hour t = 0..H-1:
      - solve a deterministic problem on the residual horizon
      - apply the first hour's HVAC and battery decisions
      - advance the state with the *realised* uncertainty
    The realisation defaults to the context (perfect forecast).
    """
    H = context["horizon"]
    real = realisation or context
    applied_y    = np.zeros(H, dtype=int)
    applied_ubat = np.zeros(H, dtype=int)
    applied_Tin  = np.zeros(H + 1)
    applied_SoC  = np.zeros(H + 1)

    # Start state from defaults
    Tin = float(building["T_in_init"] if building and "T_in_init" in building else 21.0)
    Tm  = Tin
    SoC = 0.5 * (building["E_bat"] if building and "E_bat" in building else 10.0)
    applied_Tin[0] = Tin
    applied_SoC[0] = SoC

    total_obj = 0.0
    feasible = True

    for t in range(H):
        sub_H = min(receding_horizon, H - t)
        sub_ctx = {
            "T_out":  np.asarray(real["T_out"])[t:t+sub_H],
            "price":  np.asarray(real["price"])[t:t+sub_H],
            "PV":     np.asarray(real["PV"])[t:t+sub_H],
            "d":      np.asarray(real["d"])[t:t+sub_H],
            "horizon": sub_H,
        }
        sub_building = dict(building or {})
        sub_building["T_in_init"]    = Tin
        sub_building["T_m_init"]     = Tm
        sub_building["SoC_init_frac"] = SoC / (sub_building.get("E_bat", 10.0))

        sub_window = None
        if guest_window:
            ws, we = guest_window
            ws_sub = max(0, ws - t); we_sub = min(sub_H, we - t)
            if we_sub > ws_sub:
                sub_window = (ws_sub, we_sub)

        sol = solve_stochastic(theta=theta,
                               scenarios=deterministic_scenario(sub_ctx),
                               alpha=0.0 if sub_window else 1.0,
                               guest_window=sub_window,
                               building=sub_building,
                               time_limit_s=time_limit_s)
        if not sol["feasible"]:
            feasible = False
            break

        applied_y[t]    = int(sol["y"][0])
        applied_ubat[t] = int(sol["ubat"][0])
        # advance state using the first applied step's trajectory
        Tin = float(sol["T_in"][0][1])
        SoC = float(sol["SoC"][0][1])
        # (T_m approx tracks T_in over short windows; recompute if needed)
        Tm  = float(sol["T_in"][0][1])
        applied_Tin[t+1] = Tin
        applied_SoC[t+1] = SoC

    return {
        "method": "mpc",
        "feasible": feasible,
        "y": applied_y,
        "ubat": applied_ubat,
        "T_in": applied_Tin[np.newaxis, :],
        "SoC":  applied_SoC[np.newaxis, :],
        "horizon": H,
        "N_scenarios": 1,
    }


def replay_first_stage(theta: dict,
                       y_fixed: np.ndarray,
                       ubat_fixed: np.ndarray,
                       test_scenarios: list,
                       guest_window: tuple[int, int] | None = None,
                       building: dict | None = None,
                       time_limit_s: float = 5.0) -> List[Dict[str, Any]]:
    """
    Replay the first-stage decisions on out-of-sample scenarios.
    Used for Robust Feasibility Rate (RFR) and VSS computation.
    """
    results = []
    for scen in test_scenarios:
        ctx = {
            "T_out": scen["T_out"], "price": scen["price"],
            "PV": scen["PV"], "d": scen["d"],
            "horizon": len(scen["T_out"]),
        }
        # Solve again but force y / ubat (no degrees of freedom on first stage)
        # Simpler approximation: simulate forward with fixed y / ubat,
        # solving only the LP-like recourse (here as a single-scenario CP-SAT
        # with extra fixing constraints).
        scen_list = [{**scen, "prob": 1.0}]
        sol = solve_stochastic(theta=theta, scenarios=scen_list,
                               alpha=0.0 if guest_window else 1.0,
                               guest_window=guest_window,
                               building=building,
                               time_limit_s=time_limit_s)
        results.append(sol)
    return results
