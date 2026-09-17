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
                       time_limit_s: float = 5.0,
                       joint: bool = True) -> Dict[str, Any]:
    """
    Replay a committed first-stage plan (y_fixed, ubat_fixed) on
    out-of-sample scenarios, for RFR and VSS evaluation.

    joint=True  : one solve over the whole test ensemble with the first
                  stage fixed. This is the evaluation required to bound
                  the VSS, because it returns a single objective with
                  the solver's incumbent and lower bound.
    joint=False : one solve per scenario (diagnostic only).

    The chance constraint is DISABLED (alpha=1.0) during replay: the plan
    is being evaluated, not re-optimized, so comfort outcomes must be
    observed rather than enforced. Returns the solver status so that a
    timeout (UNKNOWN) is never mistaken for proven infeasibility.
    """
    if joint:
        scen_list = [{**sc, "prob": 1.0 / len(test_scenarios)}
                     for sc in test_scenarios]
        sol = solve_stochastic(theta=theta, scenarios=scen_list,
                               alpha=1.0,
                               guest_window=guest_window,
                               building=building,
                               time_limit_s=time_limit_s,
                               fix_y=y_fixed, fix_ubat=ubat_fixed)
        return {
            "joint": True,
            "feasible": bool(sol.get("feasible")),
            "status": sol.get("status"),
            "proved_infeasible": bool(sol.get("proved_infeasible")),
            "objective": sol.get("objective"),
            "best_bound": sol.get("best_bound"),
            "objective_normalized": sol.get("objective_normalized"),
            "T_min_effective": sol.get("T_min_effective"),
            "T_in": sol.get("T_in"),
        }

    per_scenario = []
    for scen in test_scenarios:
        sol = solve_stochastic(theta=theta,
                               scenarios=[{**scen, "prob": 1.0}],
                               alpha=1.0,
                               guest_window=guest_window,
                               building=building,
                               time_limit_s=time_limit_s,
                               fix_y=y_fixed, fix_ubat=ubat_fixed)
        per_scenario.append({
            "feasible": bool(sol.get("feasible")),
            "status": sol.get("status"),
            "proved_infeasible": bool(sol.get("proved_infeasible")),
            "objective": sol.get("objective"),
            "T_min_effective": sol.get("T_min_effective"),
            "T_in": sol.get("T_in"),
        })
    return {"joint": False, "per_scenario": per_scenario}


def vss_bounds(replay: Dict[str, Any], sp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Interval for the Value of Stochastic Solution from a feasible joint
    replay of the expected-value plan and the stochastic solve:

        L_EV - U_SP  <=  VSS  <=  U_EV - L_SP

    where [L, U] are (best_bound, objective) of each solve. Returns None
    bounds when either solve did not produce both quantities.
    """
    L_ev, U_ev = replay.get("best_bound"), replay.get("objective")
    L_sp, U_sp = sp.get("best_bound"), sp.get("objective")
    ok = all(v is not None for v in (L_ev, U_ev, L_sp, U_sp)) \
        and replay.get("feasible") and sp.get("feasible")
    if not ok:
        return {"certified": False, "lower": None, "upper": None,
                "reason": "replay or stochastic solve lacks bounds "
                          "or was not feasible"}
    return {"certified": True,
            "lower": L_ev - U_sp,
            "upper": U_ev - L_sp}
