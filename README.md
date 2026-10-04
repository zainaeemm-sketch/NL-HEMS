# NL-HEMS — Linguistically Risk-Aware Home Energy Management

NL-HEMS is a conversational home energy management system. An occupant states a request in plain language ("guests are coming tonight, keep it warm but watch the bill"). The system parses it into a structured intent, asks for clarification when needed, and turns it into a day-ahead schedule for the heat pump (HVAC) and the home battery, with rooftop PV, under uncertain weather, prices, PV output and load.

The central idea is an explicit, inspectable policy **α(z)**. It maps cues in the request (guests, comfort priority, cost emphasis, hedging, medical context) to the **violation allowance of a scenario-based comfort constraint**: the largest fraction of sampled scenarios in which the indoor temperature may fall below the lower comfort bound during the requested time window. A medical request sets α = 0, so no scenario may fall below the bound.

**Interactive demo:** <https://nl-hems.streamlit.app/>

---

## Paper

> Z. Naeem, G. Cirrincione, M. Di Silvestre, R. Musca, E. Riva Sanseverino, G. Sciumè, G. Zizzo, P. Zhang.
> *Linguistically Risk-Aware Home Energy Management: A Conversational Stochastic Framework with PV and Battery Coordination.*
> Manuscript submitted to *Applied Energy* (2026). The full citation will be added on publication.

---

## How the pipeline works

```
utterance u ──parse──▶ intent z ──clarify──▶ z' ──fuzzy map Φ──▶ θ (target, lower bound, weights)
                                              └──risk policy──▶ α(z), active window W
context c + seed ──AR(1) sampler──▶ N_s scenarios
(θ, W, α, scenarios) ──two-stage stochastic CP-SAT──▶ HVAC on/off + battery mode schedule
```

1. **Parsing.** A rule-based parser and an LLM parser (with schema validation and fallback to the rules) map the utterance to a typed intent `z`.
2. **Clarification gate.** Before any schedule is computed, deterministic rules check for vague time references, conflicting comfort/cost requests, and missing required fields. If any rule fires, the system asks instead of scheduling.
3. **Fuzzy preference layer.** Look-up tables and triangular memberships turn the comfort label, priorities and guest flag into a target temperature, a lower comfort bound and objective weights.
4. **Risk policy α(z).** This is a product of one factor (modulator) per cue, clipped to a cap:
   `α(z) = 1[m=0] · clip_[0, 0.30]( 0.20 · 2^(−g) · (1 − 0.7 r) · (1 + 0.6·1[cost=high]) · (2 − ι) )`.
   The coefficients are **design defaults**, not values fitted to occupant data.
5. **Optimization.** A two-stage stochastic program on equiprobable AR(1) scenarios (2R2C thermal model, PV, battery, grid import/export) is solved with Google OR-Tools CP-SAT. HVAC on/off and battery mode are first-stage decisions; powers are scenario-dependent recourse. The comfort chance constraint is joint over the active window and allows at most `K = ⌊N_s·α⌋` violating scenarios.

---

## App pages

| Page | What it does | Used in the paper |
|---|---|---|
| 1. Overview | Summary of the framework and LLM-parser status. | — |
| 2. Single Command | End-to-end run for one utterance: parsing, clarification gate, fuzzy layer, α(z), schedule. | Illustrates Sections 4–5 |
| 3. Linguistic Benchmark | Runs the 46-utterance benchmark (11 difficulty axes) for the rule-based, simulated-LLM and real-LLM parsers. Exports `parser_predictions.csv` and `parser_run_config.csv`. | Tables 11–12, Fig. 5 |
| 4. Optimization Study | Exploratory comparison of stochastic, deterministic and MPC schedules. | Not reported (exploratory) |
| 5. Real Milan Data | Shows the representative Milan operating context: PVGIS-derived PV, F1/F2/F3 time-of-use tariff values, constructed base load. | Fig. 2 |
| 6. Sensitivity Analysis | Exploratory sweep over α and N_s. | Not reported (see note below) |
| 7. alpha(z) Mapping | Demonstration: runs example utterances (medical, guest, hedged) through the parser, fuzzy layer and α(z), followed by a short stochastic solve. | Illustrative only |
| 8. Reviewer Study | Point-forecast versus stochastic schedule with joint fixed-plan replay and certified solver bounds; hard versus relaxed comfort ablation. Exports `table3_no_mpc.csv` (legacy file name), `alpha_ablation.csv` and `run_config.csv`. | Tables 8–9 |
| 9. Converged Sweep | Warm-started version of the sensitivity sweep. Exports `sweep_converged.csv`. | Not reported (see note below) |
| 10. Benefit Study | For a chosen utterance, finds the harshest evening cold dip at which α = 0 is still feasible, then compares controllers with distinct integer budgets (α = K/N_s), optionally with scaled comfort weights. Committed schedules are evaluated out of sample on 64 independent scenarios. Exports `benefit_study_v2.csv`. | Table 10 |

The solver-based sweeps (pages 6 and 9) are not reported as results in the paper. At the time limits used, their residual optimality gaps are too large to support a trend claim. The paper instead reports the exact, solver-free effect of the scenario count on the enforced allowance.

---

## Repository layout

```
NL-HEMS/
├── app.py                    # Streamlit entry point (10 pages)
├── requirements.txt
├── .devcontainer/            # GitHub Codespaces / dev-container setup (Python 3.11)
├── data/
│   └── samples/benchmark_samples.jsonl   # small example commands
├── paper_figures/            # figures from an earlier version of the study (Feb 2026)
└── src/
    ├── parsers.py            # rule-based, simulated-LLM, LLM and direct parsers
    ├── fuzzy.py              # triangular fuzzy map, crisp map, alpha_from_intent()
    ├── scenarios.py          # AR(1) scenario generation and point-forecast scenario
    ├── optimizer.py          # two-stage stochastic CP-SAT model (solve_stochastic)
    ├── baselines.py          # deterministic and MPC baselines, joint replay, solver-bound intervals
    ├── benchmark.py          # 46-utterance benchmark with gold labels
    ├── metrics.py            # DR score, SCR, CVaR, RFR, VSS/EVPI, parser F1
    ├── pvgis.py              # cached PVGIS profile for Milan, F1/F2/F3 tariff, context builder
    ├── app.py                # earlier 7-page version of the app (not used)
    └── requirements.txt      # copy kept for the earlier version
```

---

## Running locally

Requires Python 3.11 (the dev container uses 3.11).

```bash
git clone https://github.com/zainaeemm-sketch/NL-HEMS.git
cd NL-HEMS
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>. Without an API key, the LLM parser falls back to the in-process simulated parser, and every page still runs.

### Configuring the LLM parser (optional)

API keys are read from Streamlit secrets and must **never** be committed to the repository.

- **Locally:** create `.streamlit/secrets.toml`, and add `.streamlit/secrets.toml` to `.gitignore`.
- **On Streamlit Community Cloud:** use *App settings → Secrets*.

```toml
# OpenAI or any OpenAI-compatible gateway
OPENAI_API_KEY  = "your-key"
OPENAI_BASE_URL = "https://your-gateway/v1"   # optional; omit for the OpenAI endpoint
LLM_MODEL       = "your-model-name"           # optional; code default: gpt-4o-mini

# or Anthropic
# ANTHROPIC_API_KEY = "your-key"
```

Reboot the app after changing secrets.

---

## Reproducing the paper results

- **Tables 8–9:** page *8. Reviewer Study*. **Table 10:** page *10. Benefit Study*. **Tables 11–12 and Fig. 5:** page *3. Linguistic Benchmark*. Use the settings stated in the corresponding table notes of the paper (scenario count, seed, active window, time limit, stress-context parameters).
- Pages 3, 8 and 10 export their results as CSV together with the run settings. For pages 8 and 10 this includes the solver status, incumbent objective, best bound and residual gap of every solve.
- **CP-SAT runs with a time limit and 4 parallel workers**, so incumbents can differ slightly between runs and machines. The paper's claims rest on solver bounds and on out-of-sample evaluation of committed schedules, not on single incumbent values.
- **Objectives are in the solver's internal scaled units**, not euros. Compare them only at a fixed scenario count.
- All optimization results in the paper use the **deterministic (rule-based) parser**. The real-LLM benchmark column is indicative, because the model served behind a gateway alias cannot be verified by a reader.

---

## Data and inputs

The case study is a simulation with a representative Milan-inspired operating context. It is **not** a measured household data set.

- **PV:** a cached PVGIS-derived profile for Milan (45.464° N, 9.190° E), scaled to the selected kWp. Live PVGIS retrieval is optional.
- **Tariff:** illustrative F1/F2/F3 time-of-use values (0.265 / 0.230 / 0.215 EUR/kWh). They follow the ARERA band structure but are not a verified current tariff.
- **Outdoor temperature and base load:** constructed representative profiles. Stress experiments apply a colder shift and an evening dip.
- **Uncertainty:** independent AR(1) errors per signal. Errors are correlated over time within each signal, but there is no dependence between signals.

---

## Scope and limitations

- α(z) is a transparent design policy. Its coefficients have not been calibrated with occupants.
- With N_s scenarios, at most ⌊N_s·0.30⌋ + 1 distinct violation budgets are reachable, so close chance levels can define the same optimization program.
- α = 0 enforces the comfort bound in every **sampled** scenario of the discretized model over the active window. It is not a guarantee for a real dwelling.
- Full-scenario recourse is an offline information assumption; the tool is not a causal receding-horizon controller.
- The thermal parameters are nominal, and the language benchmark is small and author-annotated.

---

## Contact

Zain Naeem — Department of Engineering, University of Palermo, Italy.
Questions and issues: please open a GitHub issue.
