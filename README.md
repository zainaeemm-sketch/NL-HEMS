# NL-HEMS v2 — Journal Revision Build

Redesigned Streamlit app for the paper *“Robust Natural-Language Home Energy Management: A Stochastic Programming Framework for HVAC, Photovoltaic, and Battery Coordination Under Uncertainty.”*

This v2 build directly addresses the four reviewer / supervisor comments:

| Concern                              | Where it lives in the app           |
|--------------------------------------|-------------------------------------|
| Benchmark too small / saturated      | Tab 3 – Expanded Benchmark (46 utterances, 11 difficulty axes) |
| Missing baseline comparisons         | Tab 4 – Stochastic vs Deterministic vs MPC + VSS |
| Use of real data                     | Tab 5 – PVGIS Milan + ARERA F1/F2/F3 |
| Integration vs novelty               | Tab 7 – Linguistic-to-chance-level mapping **α(z)** |
| Motivating optimization choices      | Tab 6 – Sensitivity sweeps over α and N_s |

## Repository layout

```
nl-hems-v2/
├── app.py                    # Streamlit entry point
├── requirements.txt
├── .streamlit/config.toml
├── README.md
└── src/
    ├── __init__.py
    ├── parsers.py            # Stub / SimulatedLLM / LLM / Direct
    ├── fuzzy.py              # Triangular fuzzy + crisp + α(z)
    ├── pvgis.py              # PVGIS client + ARERA tariff
    ├── scenarios.py          # AR(1) scenario generation
    ├── optimizer.py          # Two-stage stochastic CP-SAT
    ├── baselines.py          # Deterministic + MPC + replay
    ├── benchmark.py          # 46 utterance benchmark
    └── metrics.py            # VSS / EVPI / CVaR / RFR / parser F1
```

## Local run

```bash
git clone <your-repo>
cd nl-hems-v2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud deployment

1. Push the repo to GitHub.
2. On https://share.streamlit.io click *New app*.
3. Point at the repo, branch `main`, file `app.py`.
4. (Optional) Add secrets under *Advanced settings → Secrets*:

   ```toml
   OPENAI_API_KEY = "sk-..."
   # or
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```

   Without a key the LLM parser falls back to the in-process
   SimulatedLLMParser — the app still works fully.

## Notes for the paper revision

- The **α(z) mapping** (`src/fuzzy.py::alpha_from_intent`) is the
  promotable optimization novelty. Cite it in Section IX as a new
  contribution.
- **VSS** in Tab 4 directly answers “is stochastic actually better?”.
  Report VSS, mean CV, and CVaR_α side by side in the revised paper.
- The benchmark in `src/benchmark.py` is now linguistically
  diverse enough that parser differences emerge (paraphrastic,
  adversarial, code-switched, medical context). The previous
  saturation (100 % everywhere) does not survive the v2 benchmark.
- PVGIS cached profile is from `lat=45.464, lon=9.190` (Milan
  centre). ARERA bands are example values — update them with the
  current quarterly figures before submission.
