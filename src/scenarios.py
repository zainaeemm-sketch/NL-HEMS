"""
Scenario generation for stochastic optimization.

AR(1) noise paths on outdoor temperature, electricity price, and PV
generation per paper Eq. (24).  Independent draws produce a scenario
set Omega of size N_s.

For 'paired' scenarios used in VSS / EVPI metrics, train and test
sets are sampled with different seeds from the same generating
distribution.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List


def ar1_path(rho: float, sigma: float, horizon: int,
             rng: np.random.Generator) -> np.ndarray:
    """Generate one AR(1) sample path with stationary variance sigma^2."""
    out = np.zeros(horizon)
    eps_std = sigma * np.sqrt(max(0.0, 1.0 - rho ** 2))
    out[0] = rng.normal(0.0, sigma)
    for t in range(1, horizon):
        out[t] = rho * out[t - 1] + rng.normal(0.0, eps_std)
    return out


def generate_scenarios(context: dict,
                       N_s: int = 8,
                       rho: float = 0.6,
                       sigma_Tout: float = 1.0,
                       sigma_price: float = 0.05,
                       sigma_pv: float = 0.15,
                       sigma_load: float = 0.08,
                       seed: int = 0) -> List[Dict[str, np.ndarray]]:
    """
    Build N_s scenarios by perturbing the baseline context with
    independent AR(1) noise per signal.

    sigma_Tout : deg C    additive
    sigma_price: fraction multiplicative
    sigma_pv   : fraction multiplicative, clipped at 0
    sigma_load : fraction multiplicative, clipped at 0 (residential
                 non-HVAC base load uncertainty)
    """
    H = context["horizon"]
    rng = np.random.default_rng(seed)
    scenarios = []

    base_T = np.asarray(context["T_out"], dtype=float)
    base_p = np.asarray(context["price"], dtype=float)
    base_pv = np.asarray(context["PV"], dtype=float)
    base_load = np.asarray(context.get("load", np.full(H, 0.4)), dtype=float)

    for w in range(N_s):
        eT = ar1_path(rho, sigma_Tout, H, rng)
        ep = ar1_path(rho, sigma_price, H, rng)
        epv = ar1_path(rho, sigma_pv, H, rng)
        eload = ar1_path(rho, sigma_load, H, rng)
        scenarios.append({
            "T_out": base_T + eT,
            "price": np.maximum(0.01, base_p * (1.0 + ep)),
            "PV":    np.maximum(0.0,  base_pv * (1.0 + epv)),
            "load":  np.maximum(0.0,  base_load * (1.0 + eload)),
            "d":     np.asarray(context["d"], dtype=int),
            "prob":  1.0 / N_s,
        })
    return scenarios


def deterministic_scenario(context: dict) -> List[Dict[str, np.ndarray]]:
    """Single point-forecast scenario; used to compute VSS."""
    H = len(context["T_out"])
    return [{
        "T_out": np.asarray(context["T_out"], dtype=float),
        "price": np.asarray(context["price"], dtype=float),
        "PV":    np.asarray(context["PV"],    dtype=float),
        "load":  np.asarray(context.get("load", np.full(H, 0.4)), dtype=float),
        "d":     np.asarray(context["d"],     dtype=int),
        "prob":  1.0,
    }]
