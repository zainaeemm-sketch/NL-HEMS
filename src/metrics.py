"""
Performance indicators for the redesigned benchmark.

This module implements the four indicators of paper Section V-F:
  - Demand-response score (DRScore)
  - Self-consumption ratio (SCR)
  - Conditional-Value-at-Risk on comfort violation (CVaR_alpha)
  - Robust feasibility rate (RFR)

plus the two reviewer-requested stochastic-programming quantities:
  - Value of Stochastic Solution (VSS)
  - Expected Value of Perfect Information (EVPI)

and the language-layer metrics used in the v2 benchmark:
  - Guest accuracy, clarification F1, parser fallback rate,
    field-level micro-precision/recall.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Any


# ---------------------------------------------------------------------
# Control-side metrics
# ---------------------------------------------------------------------
def dr_score(y: np.ndarray, d: np.ndarray, y_base: np.ndarray | None = None) -> float:
    """Eq. 25. y_base default: HVAC always on during DR hours."""
    if y_base is None:
        y_base = np.ones_like(y)
    num = np.sum(d * y)
    den = np.sum(d * y_base) + 1e-9
    return float(1.0 - num / den)


def self_consumption_ratio(PV: np.ndarray, P_HVAC: float, y: np.ndarray,
                           P_load: float, P_ch: np.ndarray) -> float:
    """Eq. 26 averaged over a single trajectory."""
    consumed = np.minimum(PV, P_HVAC * y + P_load + P_ch)
    return float(np.sum(consumed) / (np.sum(PV) + 1e-9))


def comfort_violation_count(T_in: np.ndarray, T_min: float) -> int:
    """Return number of hours below T_min for one trajectory."""
    return int(np.sum(T_in[1:] < T_min - 1e-6))


def cvar_alpha(violation_per_scenario: List[int], alpha: float) -> float:
    """Eq. 27. CVaR on comfort violation counts at level alpha."""
    arr = np.asarray(violation_per_scenario, dtype=float)
    if arr.size == 0:
        return 0.0
    arr_sorted = np.sort(arr)[::-1]
    k = max(1, int(np.ceil(alpha * arr.size)))
    return float(np.mean(arr_sorted[:k]))


def robust_feasibility_rate(replay_results: List[Dict[str, Any]]) -> float:
    """Eq. 28. Fraction of out-of-sample scenarios for which the
    fixed first-stage plan remains feasible."""
    if not replay_results:
        return 0.0
    return float(np.mean([1 if r.get("feasible") else 0 for r in replay_results]))


# ---------------------------------------------------------------------
# Stochastic programming quantities (NEW for v2)
# ---------------------------------------------------------------------
def value_of_stochastic_solution(stoch_obj: float,
                                 det_obj_on_scenarios: float) -> float:
    """
    VSS = E[Q(det_solution)] - SP_objective.

    A positive VSS quantifies how much the stochastic solution improves
    over fixing first-stage decisions from the deterministic
    (point-forecast) problem.
    """
    return float(det_obj_on_scenarios - stoch_obj)


def expected_value_perfect_information(stoch_obj: float,
                                       wait_and_see_avg: float) -> float:
    """
    EVPI = SP_objective - E_omega[ min Q(omega) ]

    EVPI measures the cost of having to commit before observing
    uncertainty; an upper bound on the value of better forecasts.
    """
    return float(stoch_obj - wait_and_see_avg)


# ---------------------------------------------------------------------
# Language-layer metrics
# ---------------------------------------------------------------------
def field_match(pred: Dict[str, Any], gold: Dict[str, Any],
                fields=("guest_flag", "comfort_label", "cost_label",
                        "dr_label", "medical_context")) -> Dict[str, float]:
    """Micro-precision / recall / F1 on the structured intent fields."""
    tp = fp = fn = 0
    correct = total = 0
    for f in fields:
        gv = gold.get(f)
        pv = pred.get(f)
        if gv is None:
            continue
        total += 1
        if pv == gv:
            tp += 1; correct += 1
        else:
            if pv is not None:
                fp += 1
            fn += 1
    P = tp / (tp + fp + 1e-9)
    R = tp / (tp + fn + 1e-9)
    F = 2 * P * R / (P + R + 1e-9)
    return {"precision": P, "recall": R, "f1": F,
            "accuracy": correct / max(1, total)}


def clarification_metrics(pred_clar: List[int], gold_clar: List[int]) -> Dict[str, float]:
    """Binary precision/recall on the clarification-needed flag."""
    pred_clar = np.asarray(pred_clar); gold_clar = np.asarray(gold_clar)
    tp = int(np.sum((pred_clar == 1) & (gold_clar == 1)))
    fp = int(np.sum((pred_clar == 1) & (gold_clar == 0)))
    fn = int(np.sum((pred_clar == 0) & (gold_clar == 1)))
    P = tp / (tp + fp + 1e-9)
    R = tp / (tp + fn + 1e-9)
    F = 2 * P * R / (P + R + 1e-9)
    return {"clar_precision": P, "clar_recall": R, "clar_f1": F}


def parser_summary(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate language-layer summary across benchmark records."""
    if not records:
        return {}
    guest_correct = np.mean([r["guest_match"] for r in records])
    fallback_rate = np.mean([r.get("fallback", 0) for r in records])
    f1_micro      = np.mean([r["field_f1"] for r in records])
    clar_f1       = np.mean([r.get("clar_f1", 0.0) for r in records])
    return {
        "guest_accuracy_pct": 100.0 * guest_correct,
        "field_f1_mean":      f1_micro,
        "clar_f1_mean":       clar_f1,
        "fallback_rate_pct":  100.0 * fallback_rate,
        "n":                  len(records),
    }
