"""
Fuzzy preference layer.

Three mappings from structured intent z to numerical parameter vector theta:
  - triangular_map: Type-1 triangular fuzzy sets (paper formulation)
  - crisp_map:      Direct lookup table (baseline, used to ablate fuzzy)
  - alpha_from_intent: NOVEL linguistic-to-chance-level mapping

The alpha_from_intent function is the formal contribution that
addresses reviewer concerns about optimization novelty: standard
chance-constrained programming treats alpha as a fixed hyperparameter,
whereas here it is derived from the user's linguistic risk tolerance.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any


# ---------------------------------------------------------------------
# Canonical numeric anchors for linguistic labels
# ---------------------------------------------------------------------
T_TARGET_LOOKUP = {
    "cool":    19.0,
    "neutral": 20.5,
    "warm":    21.5,
    "hot":     23.0,
}

COST_WEIGHT_LOOKUP = {
    "low":    0.5,
    "medium": 1.5,
    "high":   4.0,
}

DR_WEIGHT_LOOKUP = {
    "low":    0.5,
    "medium": 1.5,
    "high":   4.0,
}


# ---------------------------------------------------------------------
# Triangular membership (paper Eq. 7-8)
# ---------------------------------------------------------------------
def triangular(v: float, a: float, b: float, c: float) -> float:
    """Triangular membership with support [a, c] and peak b."""
    if v <= a or v >= c:
        return 0.0
    if v <= b:
        return (v - a) / (b - a) if b > a else 1.0
    return (c - v) / (c - b) if c > b else 1.0


def fuzzy_target_temperature(label: str, intensity: float = 1.0) -> float:
    """Centroid combination across neighbouring labels weighted by intensity."""
    anchors = T_TARGET_LOOKUP
    centers = list(anchors.values())
    labels  = list(anchors.keys())
    if label not in labels:
        return 20.5
    idx = labels.index(label)
    # Build triangular fuzzy set centred on label, width = 1.0 deg C
    weights = []
    values  = []
    for i, c in enumerate(centers):
        mu = triangular(c, centers[max(0, idx-1)] - 0.5,
                          centers[idx],
                          centers[min(len(centers)-1, idx+1)] + 0.5)
        if mu > 0:
            weights.append(mu * intensity + (1.0 - intensity) * (1.0 if i == idx else 0.0))
            values.append(c)
    if not weights:
        return centers[idx]
    return float(np.dot(weights, values) / np.sum(weights))


# ---------------------------------------------------------------------
# Full mappings
# ---------------------------------------------------------------------
def triangular_map(z: Dict[str, Any]) -> Dict[str, Any]:
    """Fuzzy mapping z -> theta (paper Section IV-D)."""
    l_comf  = z.get("comfort_label", "neutral")
    r_comf  = float(z.get("comfort_priority", 0.5))
    l_cost  = z.get("cost_label", "medium")
    l_dr    = z.get("dr_label", "medium")
    g       = int(z.get("guest_flag", 0))
    intens  = float(z.get("comfort_intensity", 1.0))

    T_tar = fuzzy_target_temperature(l_comf, intens)
    # Context-aware gap between target and minimum
    delta = 1.0 if g == 1 else (2.0 if l_cost == "high" else 1.5)
    T_min = T_tar - delta

    return {
        "T_target": T_tar,
        "T_min":    T_min,
        "lambda_comf": 2.0 + 5.0 * r_comf + (3.0 if g else 0.0),
        "lambda_min":  10.0 + 5.0 * r_comf + (10.0 if g else 0.0),
        "lambda_cost": COST_WEIGHT_LOOKUP.get(l_cost, 1.5),
        "lambda_dr":   DR_WEIGHT_LOOKUP.get(l_dr, 1.5),
        "lambda_sw":   0.3,
        "guest_hard":  g == 1 and intens > 0.7,
    }


def crisp_map(z: Dict[str, Any]) -> Dict[str, Any]:
    """Direct table lookup (baseline for fuzzy ablation)."""
    l_comf = z.get("comfort_label", "neutral")
    l_cost = z.get("cost_label", "medium")
    l_dr   = z.get("dr_label", "medium")
    g      = int(z.get("guest_flag", 0))

    T_tar = T_TARGET_LOOKUP.get(l_comf, 20.5)
    T_min = T_tar - (1.0 if g == 1 else 2.0)

    return {
        "T_target": T_tar,
        "T_min":    T_min,
        "lambda_comf": 5.0 if g else 2.5,
        "lambda_min":  15.0 if g else 10.0,
        "lambda_cost": COST_WEIGHT_LOOKUP.get(l_cost, 1.5),
        "lambda_dr":   DR_WEIGHT_LOOKUP.get(l_dr, 1.5),
        "lambda_sw":   0.3,
        "guest_hard":  g == 1,
    }


# ---------------------------------------------------------------------
# NOVELTY: linguistic-to-chance-level mapping alpha(z)
# ---------------------------------------------------------------------
def alpha_from_intent(z: Dict[str, Any]) -> float:
    """
    Map structured intent z to chance-constraint level alpha in [0, 0.3].

    Standard chance-constrained programming treats alpha as a fixed
    analyst-chosen knob. Here it is read off the linguistic surface
    via the structured intent: insistent / medically-loaded utterances
    push alpha towards 0 (deterministic hard constraint), hedged
    utterances push alpha upwards (more violation budget).

    Inputs (fields of z, all optional):
      guest_flag        : 0/1, guest event
      comfort_priority  : [0, 1]
      cost_label        : low / medium / high
      comfort_intensity : [0, 1], from linguistic hedges (must=1, try=0.3)
      medical_context   : 0/1, set by parser on elderly / infant / health refs

    Returns
    -------
    alpha : float in [0, 0.3]
    """
    base = 0.20

    g = int(z.get("guest_flag", 0))
    if g:
        base *= 0.5  # guests halve violation budget

    r = float(z.get("comfort_priority", 0.5))
    base *= (1.0 - 0.7 * r)  # high comfort priority shrinks alpha

    if z.get("cost_label", "medium") == "high":
        base *= 1.6  # heavy cost emphasis allows more violations

    intensity = float(z.get("comfort_intensity", 1.0))
    base *= (2.0 - intensity)  # hedged language widens alpha

    if int(z.get("medical_context", 0)) == 1:
        base = 0.0  # medical override: hard constraint

    return float(np.clip(base, 0.0, 0.3))


# ---------------------------------------------------------------------
# Convenience: pick a mapping by name
# ---------------------------------------------------------------------
def get_mapping(name: str):
    return {"fuzzy": triangular_map, "crisp": crisp_map}.get(name, triangular_map)
