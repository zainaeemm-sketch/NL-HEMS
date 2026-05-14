"""
src/scenarios.py
================
Scenario generator for the journal-version stochastic HEMS.

The conference version of NL-HEMS-DR uses a single deterministic context.
The journal version treats the operating context as uncertain and reasons
over a small scenario tree. This module turns a baseline context (the same
one the conference solver consumes) into N_s scenarios with probabilities,
by sampling realizations of outdoor temperature, electricity price, and PV
generation from forecast-error models.

Output format is intentionally compatible with the existing `optimize_schedule`
context dict, so each scenario can be passed to the legacy solver unchanged
(sample-average approximation) or to the new stochastic solver as a batch.

Author: NL-HEMS journal extension
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import math
import random


# --------------------------------------------------------------------------- #
# Forecast-error model parameters
# --------------------------------------------------------------------------- #

@dataclass
class ForecastErrorModel:
    """
    Parameters describing the (Gaussian) forecast-error model on each
    exogenous signal. All sigmas are in the same units as the underlying
    quantity. Set sigma to 0 to disable uncertainty for that signal.
    """
    # Outdoor temperature error (°C). Modest by default; widen for stress tests.
    sigma_temp_c: float = 1.5
    # Multiplicative price error (fraction). 0.15 means ±15 % price noise.
    sigma_price_frac: float = 0.15
    # PV generation error (kW). Tied to the baseline PV peak; see sample_pv().
    sigma_pv_frac_of_peak: float = 0.20
    # AR(1) coefficient on hourly errors (0 = white noise, 0.9 = highly persistent).
    rho: float = 0.6
    # Reproducibility
    seed: Optional[int] = 42


# --------------------------------------------------------------------------- #
# Baseline forecast helpers
# --------------------------------------------------------------------------- #

def default_pv_profile(hours: List[int], peak_kw: float = 3.0) -> Dict[int, float]:
    """
    A simple bell-shaped PV profile centered around solar noon (h=13), zero
    before sunrise and after sunset. Replace with your own forecast if you
    have one.
    """
    profile: Dict[int, float] = {}
    sunrise, sunset = 6, 19
    for h in hours:
        if h < sunrise or h > sunset:
            profile[h] = 0.0
        else:
            # cosine bell over [sunrise, sunset]
            t = (h - sunrise) / max(1, (sunset - sunrise))
            profile[h] = max(0.0, peak_kw * math.sin(math.pi * t))
    return profile


def default_outdoor_temp(hours: List[int], t_min: float = 6.0, t_max: float = 12.0) -> Dict[int, float]:
    """
    Diurnal outdoor temperature centered around h=15 (warmest). Used as the
    baseline forecast on which forecast-error noise is then added.
    """
    profile: Dict[int, float] = {}
    for h in hours:
        # cosine, max at h=15
        phase = math.cos(2 * math.pi * (h - 15) / 24.0)
        profile[h] = t_min + (t_max - t_min) * (0.5 + 0.5 * phase)
    return profile


# --------------------------------------------------------------------------- #
# AR(1) noise sampler
# --------------------------------------------------------------------------- #

def _ar1_path(n: int, rho: float, sigma: float, rng: random.Random) -> List[float]:
    """Generate an n-step AR(1) zero-mean path with persistence rho and stddev sigma."""
    if n == 0:
        return []
    path = [rng.gauss(0.0, sigma)]
    for _ in range(1, n):
        eps = rng.gauss(0.0, sigma * math.sqrt(max(0.0, 1.0 - rho * rho)))
        path.append(rho * path[-1] + eps)
    return path


# --------------------------------------------------------------------------- #
# Scenario container
# --------------------------------------------------------------------------- #

@dataclass
class Scenario:
    """
    One realization of the uncertain context over the optimization horizon.

    `as_context_override` produces a context dict that is drop-in compatible
    with your existing `optimize_schedule(mapped, context=...)` call, so a
    scenario can be replayed through the legacy solver without modification.
    """
    scenario_id: int
    probability: float
    hours: List[int]
    outdoor_temp_c: Dict[int, float]            # T_out(t)
    price: Dict[int, float]                     # p(t)  (same units as baseline)
    pv_kw: Dict[int, float]                     # P_PV(t) in kW
    dr_event_hours: List[int] = field(default_factory=list)

    def as_context_override(self, baseline_context: Dict) -> Dict:
        """
        Compose a context override dict that the existing solver understands.
        Adds three new keys (`outdoor_temp_c`, `pv_kw`) that the journal solver
        consumes; the legacy solver simply ignores them.
        """
        ctx = dict(baseline_context or {})
        ctx["price"] = dict(self.price)
        ctx["dr_event_hours"] = list(self.dr_event_hours)
        ctx["outdoor_temp_c"] = dict(self.outdoor_temp_c)   # new (journal)
        ctx["pv_kw"] = dict(self.pv_kw)                     # new (journal)
        ctx["scenario_id"] = self.scenario_id
        ctx["scenario_probability"] = self.probability
        return ctx


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def generate_scenarios(
    baseline_context: Dict,
    n_scenarios: int = 5,
    error_model: Optional[ForecastErrorModel] = None,
    pv_peak_kw: float = 3.0,
) -> List[Scenario]:
    """
    Build `n_scenarios` realizations around the baseline context.

    Parameters
    ----------
    baseline_context : dict
        The same dict your sidebar produces (`override_context` in app.py).
        Must contain `horizon_start_hour`, `horizon_end_hour`, `price`,
        `dr_event_hours`, and ideally `initial_temp_c`.
    n_scenarios : int
        Number of scenarios. n=1 is equivalent to deterministic execution
        and reproduces the conference-paper result, which is useful as a
        baseline column in the journal table.
    error_model : ForecastErrorModel
        Forecast-error parameters; defaults to a moderate-uncertainty setting.
    pv_peak_kw : float
        Peak nameplate of the rooftop PV system in kW. Set to 0 to model a
        home without PV (battery-only journal experiment).

    Returns
    -------
    List[Scenario] of length n_scenarios. Probabilities sum to 1.
    """
    em = error_model or ForecastErrorModel()
    rng = random.Random(em.seed)

    h_start = int(baseline_context.get("horizon_start_hour", 16))
    h_end = int(baseline_context.get("horizon_end_hour", 24))
    hours = list(range(h_start, h_end))
    n = len(hours)

    base_temp = default_outdoor_temp(hours)
    base_pv = default_pv_profile(hours, peak_kw=pv_peak_kw)
    base_price = {int(h): float(baseline_context.get("price", {}).get(int(h), 2.0)) for h in hours}
    base_dr = list(baseline_context.get("dr_event_hours", []))

    scenarios: List[Scenario] = []
    for k in range(n_scenarios):
        # AR(1) noise paths
        eps_T = _ar1_path(n, em.rho, em.sigma_temp_c, rng)
        eps_p = _ar1_path(n, em.rho, em.sigma_price_frac, rng)
        eps_pv = _ar1_path(n, em.rho, em.sigma_pv_frac_of_peak, rng)

        T_sc, p_sc, pv_sc = {}, {}, {}
        for i, h in enumerate(hours):
            T_sc[h] = float(base_temp[h] + eps_T[i])
            # multiplicative price noise, clipped at zero
            p_sc[h] = max(0.0, float(base_price[h] * (1.0 + eps_p[i])))
            # PV noise scaled by peak, clipped at [0, 1.1 * peak]
            pv_sc[h] = max(
                0.0,
                min(1.1 * pv_peak_kw, float(base_pv[h] + eps_pv[i] * pv_peak_kw)),
            )

        scenarios.append(
            Scenario(
                scenario_id=k,
                probability=1.0 / n_scenarios,   # uniform; use clustering for non-uniform
                hours=hours,
                outdoor_temp_c=T_sc,
                price=p_sc,
                pv_kw=pv_sc,
                dr_event_hours=list(base_dr),
            )
        )

    return scenarios


# --------------------------------------------------------------------------- #
# Quick self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    baseline = {
        "horizon_start_hour": 16, "horizon_end_hour": 24,
        "price": {16: 2, 17: 3, 18: 8, 19: 9, 20: 8, 21: 5, 22: 3, 23: 2},
        "dr_event_hours": [19, 20],
        "initial_temp_c": 20.0,
    }
    scs = generate_scenarios(baseline, n_scenarios=3, pv_peak_kw=3.0)
    for s in scs:
        print(f"scenario {s.scenario_id} (pi={s.probability:.2f}):")
        print(f"  T_out range: [{min(s.outdoor_temp_c.values()):.1f}, {max(s.outdoor_temp_c.values()):.1f}] °C")
        print(f"  price range: [{min(s.price.values()):.2f}, {max(s.price.values()):.2f}]")
        print(f"  PV max: {max(s.pv_kw.values()):.2f} kW")
