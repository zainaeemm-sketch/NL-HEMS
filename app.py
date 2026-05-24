"""
NL-HEMS - Conversational Home Energy Management with Stochastic
          Optimization, Italian Tariff Integration, and Multi-Asset
          Coordination of HVAC, Photovoltaic, and Battery Storage.

Streamlit dashboard exposing each component of the framework:

  1. Overview            : architecture and methodological contributions
  2. Single Command      : end-to-end conversational pipeline
  3. Linguistic Benchmark: 46 utterances across 11 difficulty axes
  4. Optimization Study  : Stochastic vs. Deterministic vs. MPC + VSS
  5. Real Milan Data     : PVGIS + ARERA F1/F2/F3 + outdoor temperature
  6. Sensitivity Analysis: parameter sweeps on alpha and N_s
  7. Novel Contribution  : linguistic-to-chance-level mapping alpha(z)
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.parsers     import StubParser, SimulatedLLMParser, LLMParser, DirectParser
from src.fuzzy       import triangular_map, crisp_map, alpha_from_intent
from src.pvgis       import real_milan_context, milan_pv_profile, arera_price_profile
from src.scenarios   import generate_scenarios, deterministic_scenario
from src.optimizer   import solve_stochastic
from src.baselines   import solve_deterministic, solve_mpc, replay_first_stage
from src.benchmark   import BENCHMARK, DIFFICULTY_NAMES, benchmark_summary
from src.metrics     import (dr_score, self_consumption_ratio,
                             comfort_violation_count, cvar_alpha,
                             robust_feasibility_rate,
                             value_of_stochastic_solution,
                             field_match, clarification_metrics,
                             parser_summary)


st.set_page_config(page_title="NL-HEMS",
                   layout="wide", initial_sidebar_state="expanded")

# =============================================================
# Publication-ready chart styling
# Plotly defaults (2px lines, light-blue secondary color) wash out
# when shrunk to journal column width. These overrides produce
# print-ready output that survives downscaling to ~8 cm wide.
# =============================================================
PRINT_PALETTE = [
    "#1f3864",  # deep navy        — primary series
    "#c00000",  # dark red         — secondary series
    "#2d6a4f",  # forest green     — tertiary series
    "#cc6600",  # burnt orange     — quaternary series
    "#5b2c6f",  # deep purple      — quinary series
    "#8b4513",  # saddle brown     — senary series
]
PRINT_MARKERS = ["circle", "square", "diamond", "triangle-up",
                 "triangle-down", "x"]


def style_for_print(fig, line_width: float = 3.5,
                    marker_size: int = 11,
                    show_markers: bool = True):
    """Apply publication-ready styling to a plotly figure.

    Increases line width, enlarges markers, removes the default light-
    blue secondary color, and bumps font sizes so the chart remains
    legible when shrunk to a single-column figure in IEEEtran.
    """
    fig.update_traces(
        line=dict(width=line_width),
        marker=dict(size=marker_size,
                    line=dict(width=1.2, color="white")),
        selector=dict(type="scatter")
    )
    if show_markers:
        # Force lines+markers on any line traces that started life as
        # pure lines (px.line without markers=True)
        for trace in fig.data:
            if getattr(trace, "mode", None) == "lines":
                trace.mode = "lines+markers"
    fig.update_layout(
        font=dict(size=14, family="Arial, sans-serif", color="#222"),
        title_font=dict(size=15, color="#111"),
        legend=dict(font=dict(size=13),
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="rgba(0,0,0,0.15)", borderwidth=1),
        xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12),
                   showgrid=True, gridcolor="rgba(0,0,0,0.08)",
                   linecolor="#333", linewidth=1.2,
                   ticks="outside", ticklen=5),
        yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12),
                   showgrid=True, gridcolor="rgba(0,0,0,0.08)",
                   linecolor="#333", linewidth=1.2,
                   ticks="outside", ticklen=5),
        plot_bgcolor="white",
        margin=dict(l=70, r=30, t=55, b=60),
    )
    return fig


# =====================================================================
# Sidebar - global controls
# =====================================================================
with st.sidebar:
    st.markdown("## NL-HEMS")
    st.caption("Conversational HEMS with stochastic optimization")

    tab = st.radio("Navigation",
                   ["1. Overview",
                    "2. Single Command",
                    "3. Linguistic Benchmark",
                    "4. Optimization Study",
                    "5. Real Milan Data",
                    "6. Sensitivity Analysis",
                    "7. alpha(z) Mapping"],
                   index=0)

    st.divider()
    st.markdown("### Global building parameters")
    peak_kwp = st.number_input("PV peak (kWp)", 0.0, 15.0, 3.0, 0.5)
    E_bat    = st.number_input("Battery capacity (kWh)", 0.0, 30.0, 10.0, 1.0)
    P_bat    = st.number_input("Battery power (kW)", 0.0, 15.0, 5.0, 0.5)
    horizon  = st.number_input("Horizon (h)", 6, 48, 24, 6)


# =====================================================================
# Helper: shared context builder
# =====================================================================
@st.cache_data(show_spinner=False)
def _real_context(horizon, peak_kwp, weekday=2, try_live=False):
    return real_milan_context(horizon=horizon, weekday=weekday,
                              peak_kwp=peak_kwp, try_live=try_live)


def _building(E_bat=10.0, P_bat=5.0):
    return {"E_bat": E_bat, "P_bat": P_bat}


# =====================================================================
# TAB 1 - Overview
# =====================================================================
def tab_overview():
    st.title("NL-HEMS: Conversational Home Energy Management")
    st.markdown("*Stochastic coordination of HVAC, photovoltaic, and "
                "battery storage with natural-language intent parsing.*")

    # ---------- LLM API status banner ----------
    llm = LLMParser()
    status = llm.status()
    if status["client_ready"]:
        st.success(
            f"LLM API connected. "
            f"Backend: **{status['backend']}**  "
            f"Model: **{status['model']}**  "
            + (f"Base URL: `{status['base_url']}`"
               if status['base_url'] else "Base URL: default")
        )
        # Live ping so the user knows the API actually responds
        with st.expander("Test the API now"):
            if st.button("Send a test prompt"):
                test = llm.parse("Guests coming tonight, keep it warm but watch the bill.")
                t_status = llm.status()
                if t_status["last_error"]:
                    st.error(f"API call failed: {t_status['last_error']}")
                    st.info("The simulated fallback was used instead. "
                            "Common causes: invalid model name for this "
                            "proxy, expired key, network block, or the "
                            "proxy not supporting the chat-completions "
                            "schema. Verify the model name in your "
                            "VectorEngine dashboard matches LLM_MODEL.")
                else:
                    st.success(f"API responded. parser_name = `{test.parser_name}`, "
                               f"fallback = `{test.fallback}`")
                    st.json(test.to_dict())
    else:
        st.warning(
            "No LLM API key detected in Streamlit secrets, or the client "
            "failed to initialise. The 'LLM' parser transparently falls "
            "back to the in-process SimulatedLLMParser."
        )
        if status["last_error"]:
            st.error(f"Initialisation error: {status['last_error']}")
        st.markdown("""
**To enable a real LLM**, set the following in Streamlit
*Settings -> Secrets* (TOML format):

```toml
OPENAI_API_KEY  = "sk-..."
LLM_MODEL       = "gpt-4o-mini"                  # or your proxy's model name
OPENAI_BASE_URL = "https://api.vectorengine.ai/v1"  # only for non-OpenAI proxies
```

Restart the app from *Manage app -> Reboot* after changing secrets.
""")
    st.caption("Tip: when you run the *Single Command* tab, the parsed-intent "
               "JSON shows the actual `parser_name` used: "
               "`llm` = real API call succeeded, `llm-sim-fallback` = no key, "
               "`llm` with `fallback: true` = API call failed mid-request.")
    st.divider()

    st.markdown("""
This dashboard exposes each component of the NL-HEMS framework so the
end-to-end behaviour, the underlying optimization model, and the
language-understanding layer can be inspected separately.
""")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Framework capabilities")
        st.markdown("""
| Capability | Demonstrated in tab | Implementation |
|---|---|---|
| Linguistic robustness across paraphrastic, conflicting, code-switched, and adversarial inputs | **3. Linguistic Benchmark** | 46 utterances stratified across 11 difficulty axes |
| Optimization-method comparison and value of stochastic solution | **4. Optimization Study** | Stochastic CP-SAT vs. Deterministic MILP vs. MPC, with VSS computed |
| Real Italian residential operating context | **5. Real Milan Data** | PVGIS irradiance + ARERA F1/F2/F3 tariff bands + Milan climatology |
| Linguistic risk tolerance drives the chance-constraint level | **7. alpha(z) Mapping** | Novel linguistic-to-chance-level function alpha(z) |
| Parameter sensitivity and tractability characterisation | **6. Sensitivity Analysis** | Sweeps over alpha and scenario count N_s |
        """)

    with col2:
        st.subheader("Architecture")
        st.markdown("""
```
utterance u
   |  fparse         -> Stub / LLM / Simulated-LLM (with schema fallback)
   v
intent z (typed)
   |  fclar          -> ask-vs-act policy (Eq. 4)
   v
intent z'
   |  Phi(z)         -> Fuzzy / Crisp / Direct mapping (Eq. 6-8)
   v             |   alpha(z)  NOVEL linguistic -> chance level
parameters theta, alpha
   |  fscen          -> AR(1) scenario tree (Eq. 24)
   v
{omega}
   |  fopt           -> Stochastic CP-SAT or Deterministic / MPC baseline
   v
schedule x
```
""")
        st.info("Real Italian tariff and PVGIS data are loaded in tab 5; "
                "you can re-run any tab with the **Real Milan Data** context "
                "by enabling the toggle there.")


# =====================================================================
# TAB 2 - Single Command (original UX, refactored)
# =====================================================================
def tab_single():
    st.title("Single command - end-to-end pipeline")
    parser_name = st.selectbox("Parser", ["LLM (or simulated-LLM fallback)",
                                          "Stub (deterministic)",
                                          "Direct (no fuzzy, no schema)"],
                               index=0)
    map_name = st.selectbox("Mapping", ["Triangular fuzzy",
                                        "Crisp lookup"], index=0)
    enforce_hard = st.checkbox("Hard guest comfort (alpha derived from intent)",
                               value=True)
    N_s = st.slider("Scenarios N_s", 1, 32, 8)

    utterance = st.text_input("Utterance",
        value="Guests coming over tonight, keep it warm but watch the bill.")

    if st.button("Run pipeline", type="primary"):
        ctx = _real_context(horizon, peak_kwp)

        # Parse
        if parser_name.startswith("LLM"):
            parser = LLMParser()
        elif parser_name.startswith("Stub"):
            parser = StubParser()
        else:
            parser = DirectParser()

        if isinstance(parser, DirectParser):
            theta = parser.parse_to_theta(utterance)
            intent_dict = {"_parser": "direct"}
        else:
            intent = parser.parse(utterance)
            intent_dict = intent.to_dict()

            if map_name.startswith("Triangular"):
                theta = triangular_map(intent_dict)
            else:
                theta = crisp_map(intent_dict)

            theta["_intent"] = intent_dict

        alpha = alpha_from_intent(intent_dict) if enforce_hard else 1.0

        # Scenarios
        scens = generate_scenarios(ctx, N_s=N_s, seed=42)

        # Guest window
        gw = None
        if intent_dict.get("guest_flag") == 1:
            ws = intent_dict.get("window_start") or 18
            we = intent_dict.get("window_end") or 23
            gw = (int(ws), int(we))

        with st.spinner("Solving stochastic CP-SAT..."):
            t0 = time.time()
            sol = solve_stochastic(theta=theta, scenarios=scens, alpha=alpha,
                                   guest_window=gw,
                                   building=_building(E_bat, P_bat),
                                   time_limit_s=20)
            sol["elapsed_s"] = time.time() - t0

        st.subheader("Parsed intent / parameters")
        c1, c2 = st.columns(2)
        with c1:
            st.json(intent_dict)
        with c2:
            st.json({k: (round(v, 3) if isinstance(v, float) else v)
                     for k, v in theta.items() if not k.startswith("_")})
            st.metric("alpha (chance level)", f"{alpha:.3f}",
                      help="Derived from the linguistic intent via alpha_from_intent")

        st.subheader("Schedule")
        if not sol["feasible"]:
            st.error(f"Infeasible: {sol['status']}")
            return

        df = pd.DataFrame({
            "hour": np.arange(horizon),
            "HVAC y": sol["y"],
            "Battery mode (1=charge)": sol["ubat"],
            "T_in mean": sol["T_in"].mean(axis=0)[1:],
            "Price (EUR/kWh)": ctx["price"],
        })
        fig_t = px.line(df, x="hour", y="T_in mean",
                        labels={"hour": "Hour (h)",
                                "T_in mean": "Indoor temperature T_in (\u00b0C)"},
                        title="Indoor temperature trajectory (scenario mean)",
                    color_discrete_sequence=PRINT_PALETTE)
        style_for_print(fig_t)
        fig_t.add_hline(y=theta["T_min"], line_dash="dot",
                        line_color="firebrick",
                        annotation_text=f"T_min = {theta['T_min']:.1f} \u00b0C")
        st.plotly_chart(fig_t, use_container_width=True)

        df_ctrl = df.melt(id_vars="hour",
                          value_vars=["HVAC y", "Battery mode (1=charge)"],
                          var_name="Signal", value_name="State")
        fig_c = px.bar(df_ctrl, x="hour", y="State", color="Signal",
                       barmode="group",
                       labels={"hour": "Hour (h)",
                               "State": "Control state y(t), u_bat(t) (\u2014)"},
                       title="HVAC and battery control schedule",
                    color_discrete_sequence=PRINT_PALETTE)
        style_for_print(fig_c)
        st.plotly_chart(fig_c, use_container_width=True)
        st.dataframe(df, hide_index=True, use_container_width=True)

        st.subheader("Indicators")
        cv = [comfort_violation_count(sol["T_in"][w], theta["T_min"])
              for w in range(sol["N_scenarios"])]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Objective", f"{sol['objective']:.0f}")
        m2.metric("Mean comfort violation (h)",
                  f"{np.mean(cv):.2f}")
        m3.metric(f"CVaR_{alpha:.2f} (h)",
                  f"{cvar_alpha(cv, max(alpha, 0.1)):.2f}")
        m4.metric("Solve time (s)", f"{sol['elapsed_s']:.2f}")


# =====================================================================
# TAB 3 - Expanded Benchmark
# =====================================================================
def tab_benchmark():
    st.title("Linguistic benchmark")
    st.markdown("46 utterances stratified across 11 difficulty axes.")

    summary = pd.DataFrame(benchmark_summary())
    fig_dist = px.bar(summary, x="name", y="n",
                      labels={"name": "Difficulty axis",
                              "n": "Number of utterances (\u2014)"},
                      title="Benchmark composition by difficulty axis",
                    color_discrete_sequence=PRINT_PALETTE)
    style_for_print(fig_dist)
    fig_dist.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_dist, use_container_width=True)

    parsers = {
        "Stub":          StubParser(),
        "Simulated-LLM": SimulatedLLMParser(),
        "LLM (real or simulated fallback)": LLMParser(),
    }

    # Show LLM status prominently on this tab too, so users don't have
    # to switch to Overview to find out their key isn't wired up.
    _bench_llm = parsers["LLM (real or simulated fallback)"]
    _bench_status = _bench_llm.status()
    if _bench_status["client_ready"]:
        st.success(
            f"\u2705 Real LLM ready \u2014 backend "
            f"**{_bench_status['backend']}**, model "
            f"**{_bench_status['model']}**"
            + (f", base URL `{_bench_status['base_url']}`"
               if _bench_status['base_url'] else "")
        )
    else:
        st.warning(
            "\u26A0\uFE0F  Real LLM client **not configured**. The "
            "\"LLM (real or simulated fallback)\" row in the results "
            "will be the simulated parser silently impersonating the "
            "real one. Set `OPENAI_API_KEY` (and optionally "
            "`OPENAI_BASE_URL`, `LLM_MODEL`) in Streamlit secrets and "
            "reload."
            + (f"\n\nInitialisation error: `{_bench_status['last_error']}`"
               if _bench_status['last_error'] else "")
        )

    chosen = st.multiselect("Parsers to evaluate",
                            list(parsers.keys()),
                            default=list(parsers.keys()))

    diff_filter = st.multiselect("Restrict to difficulty axes",
                                 list(DIFFICULTY_NAMES.keys()),
                                 default=list(DIFFICULTY_NAMES.keys()))

    if st.button("Run language-layer benchmark", type="primary"):
        rows = []
        for parser_name in chosen:
            parser = parsers[parser_name]
            recs = []
            pred_clar, gold_clar = [], []
            for item in BENCHMARK:
                if item.difficulty not in diff_filter:
                    continue
                pred = parser.parse(item.text).to_dict()
                fm = field_match(pred, item.gold)
                guest_match = int(pred.get("guest_flag") == item.gold.get("guest_flag"))
                recs.append({
                    "parser":     parser_name,
                    "id":         item.sid,
                    "difficulty": DIFFICULTY_NAMES[item.difficulty],
                    "text":       item.text,
                    "field_f1":   fm["f1"],
                    "guest_match": guest_match,
                    "fallback":   1 if pred.get("fallback") else 0,
                    "predicted_guest": pred.get("guest_flag"),
                    "gold_guest":     item.gold.get("guest_flag"),
                    "predicted_comfort": pred.get("comfort_label"),
                    "gold_comfort": item.gold.get("comfort_label"),
                })
                pred_clar.append(int(pred.get("clarification_needed", 0)))
                gold_clar.append(int(item.needs_clarification))

            clar = clarification_metrics(pred_clar, gold_clar)
            for r in recs:
                r["clar_f1"] = clar["clar_f1"]
            rows.extend(recs)

        df = pd.DataFrame(rows)
        st.subheader("Per-utterance results")
        st.dataframe(df, hide_index=True, use_container_width=True)

        st.subheader("Parser-level summary")
        agg = (df.groupby("parser").agg(
                    field_f1=("field_f1", "mean"),
                    guest_acc=("guest_match", "mean"),
                    fallback=("fallback", "mean"),
                    clar_f1=("clar_f1", "first"),
                    n=("id", "count"))
                 .round(3))
        agg["guest_acc"]  *= 100
        agg["fallback"]   *= 100
        st.dataframe(agg)

        # Loud warning if the "real LLM" parser ran but fell back to
        # simulated on every single call. The numbers in this case are
        # NOT real LLM numbers; they are simulated-parser numbers.
        llm_row = agg.index[agg.index.str.contains("LLM \\(real")]
        if len(llm_row) > 0 and agg.loc[llm_row[0], "fallback"] >= 99:
            last_err = _bench_status.get("last_error") or \
                       getattr(_bench_llm, "_last_error", None) or \
                       "no error captured \u2014 see Streamlit logs"
            st.error(
                "\U0001F6A8 **Every real-LLM call fell back to the "
                "simulated parser** (fallback rate "
                f"{agg.loc[llm_row[0], 'fallback']:.0f}%). The numbers "
                "in the \"LLM (real or simulated fallback)\" row are "
                "the simulated parser, not the real LLM. Diagnose "
                "before using them in the paper.\n\n"
                f"Last captured error: `{last_err}`"
            )
        elif len(llm_row) > 0 and agg.loc[llm_row[0], "fallback"] > 0:
            st.warning(
                f"\u26A0\uFE0F  Real-LLM partial fallback: "
                f"{agg.loc[llm_row[0], 'fallback']:.0f}% of calls fell "
                "back to the simulated parser. Reported numbers are a "
                "mix of real and simulated."
            )

        st.subheader("Per-difficulty breakdown")
        per_diff = (df.groupby(["parser", "difficulty"])
                      .agg(field_f1=("field_f1", "mean"),
                           n=("id", "count"))
                      .reset_index())
        fig = px.bar(per_diff, x="difficulty", y="field_f1",
                     color="parser", barmode="group",
                     labels={"difficulty": "Difficulty axis",
                             "field_f1": "Field-level F1 score (\u2014)",
                             "parser": "Parser"},
                     title="Field-level F1 by difficulty axis",
                    color_discrete_sequence=PRINT_PALETTE)
        style_for_print(fig)
        fig.update_layout(yaxis_range=[0, 1.05], xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# TAB 4 - Baseline comparison + VSS
# =====================================================================
def tab_baselines():
    st.title("Optimization study: Stochastic vs. Deterministic vs. MPC")
    st.markdown("""
For a fixed user intent, three controllers are solved on the same
operating context:

  - **Stochastic** : two-stage CP-SAT over N_s scenarios with chance-constrained comfort
  - **Deterministic** : single-point-forecast MILP
  - **MPC** : receding-horizon deterministic re-solve at each hour

The **Value of Stochastic Solution (VSS)** quantifies the cost gap
that the stochastic formulation closes relative to fixing first-stage
decisions from the deterministic point-forecast problem.
""")

    utterance = st.text_input("Utterance",
        value="Guests staying late, keep it warm but watch the bill.")

    N_s = st.slider("Stochastic scenarios N_s", 4, 24, 8)
    alpha = st.slider("alpha (chance level)", 0.0, 0.5, 0.2, 0.05)

    if st.button("Run all three optimizers", type="primary"):
        ctx = _real_context(horizon, peak_kwp)

        parser = SimulatedLLMParser()
        intent = parser.parse(utterance).to_dict()
        theta  = triangular_map(intent)
        gw = None
        if intent.get("guest_flag") == 1:
            ws = intent.get("window_start") or 19
            we = intent.get("window_end") or 23
            gw = (int(ws), int(we))

        scens = generate_scenarios(ctx, N_s=N_s, seed=42)

        with st.spinner("Solving stochastic..."):
            t0 = time.time()
            sol_s = solve_stochastic(theta, scens, alpha=alpha,
                                     guest_window=gw,
                                     building=_building(E_bat, P_bat))
            t_s = time.time() - t0

        with st.spinner("Solving deterministic..."):
            t0 = time.time()
            sol_d = solve_deterministic(theta, ctx, guest_window=gw,
                                        building=_building(E_bat, P_bat))
            t_d = time.time() - t0

        with st.spinner("Running MPC (12h receding)..."):
            t0 = time.time()
            sol_m = solve_mpc(theta, ctx, guest_window=gw,
                              building=_building(E_bat, P_bat),
                              receding_horizon=12)
            t_m = time.time() - t0

        # Proper VSS computation: take the deterministic plan's
        # first-stage decisions (y, u_bat) and re-evaluate them on the
        # stochastic scenario set. The resulting expected cost is the
        # quantity to subtract from the stochastic optimum.
        if sol_d["feasible"]:
            with st.spinner("Evaluating deterministic plan on stochastic scenarios (for VSS)..."):
                sol_d_eval = solve_stochastic(
                    theta=theta, scenarios=scens, alpha=alpha,
                    guest_window=gw,
                    building=_building(E_bat, P_bat),
                    fix_y=sol_d["y"], fix_ubat=sol_d["ubat"],
                    time_limit_s=15)
        else:
            sol_d_eval = {"feasible": False, "objective": None}

        # Proper VSS: E[cost(det_plan)] - SP_opt. Positive => stochastic wins.
        if sol_s["feasible"] and sol_d_eval["feasible"]:
            vss = value_of_stochastic_solution(
                sol_s["objective"], sol_d_eval["objective"])
        else:
            vss = None

        c1, c2, c3 = st.columns(3)
        c1.metric("Stochastic objective",
                  f"{sol_s['objective']:.0f}" if sol_s["feasible"] else "infeasible",
                  delta=f"{t_s:.1f} s")
        c2.metric("Deterministic objective",
                  f"{sol_d['objective']:.0f}" if sol_d["feasible"] else "infeasible",
                  delta=f"{t_d:.1f} s")
        c3.metric("MPC feasibility",
                  "feasible" if sol_m["feasible"] else "broke",
                  delta=f"{t_m:.1f} s")

        st.subheader("Value of Stochastic Solution")
        if vss is not None:
            colA, colB, colC = st.columns(3)
            colA.metric("Stochastic objective",
                        f"{sol_s['objective']:.0f}")
            colB.metric("Det. plan eval'd on scenarios",
                        f"{sol_d_eval['objective']:.0f}")
            colC.metric("VSS", f"{vss:.0f}",
                        delta="positive = stochastic wins" if vss > 0
                              else "non-positive = stochastic ties det")
            if vss > 0:
                st.success(
                    f"VSS = {vss:.1f}. "
                    "The stochastic formulation strictly reduces the expected "
                    "cost relative to applying the deterministic plan on the "
                    "same scenario set.")
            else:
                st.info(
                    f"VSS = {vss:.1f}. "
                    "On this instance the stochastic optimum coincides with "
                    "the deterministic plan when evaluated against the "
                    "scenario set. Increase N_s, sigma, or alpha to surface "
                    "the gap.")
        else:
            st.warning("VSS not computable (one of the problems was infeasible)")

        # Indoor temperature comparison
        st.subheader("Indoor temperature trajectories")
        rows = []
        for label, sol in [("Stochastic mean", sol_s),
                           ("Deterministic", sol_d),
                           ("MPC", sol_m)]:
            if not sol.get("feasible"):
                continue
            T = sol["T_in"].mean(axis=0)
            rows.extend([{"hour": t, "T_in (C)": T[t], "method": label}
                         for t in range(len(T))])
        if rows:
            dft = pd.DataFrame(rows)
            fig = px.line(dft, x="hour", y="T_in (C)", color="method",
                          labels={"hour": "Hour (h)",
                                  "T_in (C)": "Indoor temperature T_in (\u00b0C)",
                                  "method": "Controller"},
                          title="Indoor temperature trajectories \u2014 three controllers",
                    color_discrete_sequence=PRINT_PALETTE)
            style_for_print(fig)
            fig.add_hline(y=theta["T_min"], line_dash="dot",
                          line_color="firebrick",
                          annotation_text=f"T_min = {theta['T_min']:.1f} \u00b0C")
            st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# TAB 5 - Real Milan Data (PVGIS + ARERA)
# =====================================================================
def tab_realdata():
    st.title("Real Milan data: PVGIS + ARERA F1/F2/F3 tariff + residential load")
    st.markdown("""
This tab grounds the case study in **real Italian residential operating
conditions for Milan**, replacing the synthetic profiles common in
HEMS prototypes:

  - **PV generation**: PVGIS hourly product for lat 45.464, lon 9.190
  - **Electricity price**: ARERA residential F1/F2/F3 bands
  - **Outdoor temperature**: Milan climatology
  - **Non-HVAC base load**: representative Italian residential consumption
    shape with characteristic morning and evening peaks, calibrated to
    typical family daily consumption (~8 kWh/day, excluding HVAC and EV)
  - **DR events**: 19h-20h evening peak (typical Italian residential)

If the deployment environment allows outbound HTTPS, you can try a
live PVGIS fetch; otherwise the bundled cached profile is used.
""")
    weekday = st.selectbox("Day type", ["Weekday (Mon-Fri)", "Saturday", "Sunday"],
                           index=0)
    weekday_idx = {"Weekday (Mon-Fri)": 2, "Saturday": 5, "Sunday": 6}[weekday]
    try_live = st.checkbox("Try live PVGIS fetch", value=False)

    ctx = real_milan_context(horizon=horizon, weekday=weekday_idx,
                             peak_kwp=peak_kwp, try_live=try_live)

    # Defensive sizing: use the smallest common length so DataFrame
    # construction never fails on a length mismatch.
    arrays = {
        "T_out (C)":        np.asarray(ctx["T_out"]).ravel(),
        "Price (EUR/kWh)":  np.asarray(ctx["price"]).ravel(),
        "PV (kW)":          np.asarray(ctx["PV"]).ravel(),
        "Load (kW)":        np.asarray(ctx["load"]).ravel(),
        "DR event":         np.asarray(ctx["d"]).ravel(),
    }
    H = min(int(horizon), *(len(a) for a in arrays.values()))
    arrays = {k: v[:H] for k, v in arrays.items()}

    st.caption(f"Data source: {ctx['source']}")
    df = pd.DataFrame({
        "hour":  np.arange(H),
        **arrays,
    })

    # Headline indicators
    daily_load_kwh = float(arrays["Load (kW)"].sum())
    daily_pv_kwh   = float(arrays["PV (kW)"].sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Daily non-HVAC load",       f"{daily_load_kwh:.2f} kWh")
    m2.metric("Daily PV generation",       f"{daily_pv_kwh:.2f} kWh")
    m3.metric("PV / load ratio",
              f"{daily_pv_kwh / max(1e-6, daily_load_kwh):.2f}")

    c1, c2 = st.columns(2)
    with c1:
        fig_T = px.line(df, x="hour", y="T_out (C)",
                        labels={"hour": "Hour (h)",
                                "T_out (C)": "Outdoor temperature T_out (\u00b0C)"},
                        title="Milan outdoor temperature",
                    color_discrete_sequence=PRINT_PALETTE)
        style_for_print(fig_T)
        st.plotly_chart(fig_T, use_container_width=True)

        fig_p = px.bar(df, x="hour", y="Price (EUR/kWh)",
                       labels={"hour": "Hour (h)",
                               "Price (EUR/kWh)": "Electricity price p(t) (EUR/kWh)"},
                       title="ARERA F1/F2/F3 residential tariff",
                    color_discrete_sequence=PRINT_PALETTE)
        style_for_print(fig_p)
        st.plotly_chart(fig_p, use_container_width=True)
    with c2:
        fig_pv = px.area(df, x="hour", y="PV (kW)",
                         labels={"hour": "Hour (h)",
                                 "PV (kW)": "PV generation P^PV(t) (kW)"},
                         title="PVGIS hourly PV generation (Milan)",
                    color_discrete_sequence=PRINT_PALETTE)
        style_for_print(fig_pv)
        st.plotly_chart(fig_pv, use_container_width=True)

        fig_ld = px.line(df, x="hour", y="Load (kW)",
                         labels={"hour": "Hour (h)",
                                 "Load (kW)": "Non-HVAC base load P^load(t) (kW)"},
                         title="Italian residential consumption profile",
                    color_discrete_sequence=PRINT_PALETTE)
        style_for_print(fig_ld)
        st.plotly_chart(fig_ld, use_container_width=True)
    st.dataframe(df, hide_index=True, use_container_width=True)


# =====================================================================
# TAB 6 - Sensitivity sweep
# =====================================================================
def tab_sensitivity():
    st.title("Sensitivity sweep")
    st.markdown("""
Justify the optimization choices by showing how outcomes move with
the key parameters. The sweep below uses a fixed utterance and
varies the **chance level alpha** and **scenario count N_s**.
""")

    utterance = st.text_input("Utterance",
        value="Keep us warm for guests, but help during peak hours.")
    alphas = st.multiselect("alpha values",
                            [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
                            default=[0.0, 0.1, 0.2, 0.3])
    N_list = st.multiselect("N_s values", [4, 6, 8, 12, 16, 24],
                            default=[4, 8, 16])
    tmin_override = st.slider("T_min for the sweep (°C)",
                              min_value=20.0, max_value=23.0,
                              value=22.0, step=0.5,
                              help="Higher T_min makes the chance "
                                   "constraint actually bind, so the "
                                   "violation plot becomes informative. "
                                   "Reviewers expect a non-trivial "
                                   "violation–vs–\u03b1 curve.")

    if st.button("Run sweep", type="primary"):
        ctx = _real_context(horizon, peak_kwp)
        intent = SimulatedLLMParser().parse(utterance).to_dict()
        theta  = triangular_map(intent)
        theta["T_min"] = float(tmin_override)
        gw = None
        if intent.get("guest_flag") == 1:
            ws = intent.get("window_start") or 19
            we = intent.get("window_end") or 23
            gw = (int(ws), int(we))

        rows = []
        prog = st.progress(0.0)
        total = len(alphas) * len(N_list)
        k = 0
        for a in alphas:
            for N in N_list:
                scens = generate_scenarios(ctx, N_s=N, seed=42)
                t0 = time.time()
                sol = solve_stochastic(theta, scens, alpha=a,
                                       guest_window=gw,
                                       building=_building(E_bat, P_bat),
                                       time_limit_s=15)
                cv = ([comfort_violation_count(sol["T_in"][w], theta["T_min"])
                       for w in range(sol["N_scenarios"])]
                      if sol.get("feasible") else [None])
                rows.append({
                    "alpha": a, "N_s": N,
                    "feasible": sol["feasible"],
                    "objective": sol.get("objective"),
                    "wall_s":    time.time() - t0,
                    "mean_CV":   float(np.mean([x for x in cv if x is not None]))
                                  if any(x is not None for x in cv) else None,
                    "max_CV":    float(np.max([x for x in cv if x is not None]))
                                  if any(x is not None for x in cv) else None,
                })
                k += 1
                prog.progress(k / total)
        prog.empty()

        df = pd.DataFrame(rows)
        st.subheader("Sweep results")
        st.dataframe(df, hide_index=True, use_container_width=True)
        if not df.empty:
            fig1 = px.line(df, x="alpha", y="objective", color="N_s",
                           markers=True,
                           labels={"alpha": "Chance level \u03b1 (\u2014)",
                                   "objective": "Objective J (\u2014)",
                                   "N_s": "Scenarios N_s (\u2014)"},
                           title="Stochastic objective vs chance level \u03b1",
                    color_discrete_sequence=PRINT_PALETTE)
            style_for_print(fig1)
            fig2 = px.line(df, x="alpha", y="mean_CV", color="N_s",
                           markers=True,
                           labels={"alpha": "Chance level \u03b1 (\u2014)",
                                   "mean_CV": "Mean comfort violation (h)",
                                   "N_s": "Scenarios N_s (\u2014)"},
                           title="Mean comfort violation vs chance level \u03b1",
                    color_discrete_sequence=PRINT_PALETTE)
            style_for_print(fig2)
            fig3 = px.line(df, x="N_s", y="wall_s", color="alpha",
                           markers=True,
                           labels={"N_s": "Number of scenarios N_s (\u2014)",
                                   "wall_s": "Solve time (s)",
                                   "alpha": "Chance level \u03b1 (\u2014)"},
                           title="CP-SAT solve time vs scenario count",
                    color_discrete_sequence=PRINT_PALETTE)
            style_for_print(fig3)
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)
            st.plotly_chart(fig3, use_container_width=True)


# =====================================================================
# TAB 7 - Novelty: alpha(z)
# =====================================================================
def tab_novelty():
    st.title("Linguistic-to-chance-level mapping  alpha(z)")
    st.markdown("""
Standard chance-constrained programming treats **alpha** as a fixed
analyst-chosen hyperparameter. In this framework alpha is derived from
the user's linguistic intent: insistent utterances and medical
contexts push alpha towards 0 (deterministic hard constraint);
hedged utterances expand the violation budget.

Below, three utterances of varying linguistic intensity feed through
the parser, the fuzzy mapping, and `alpha_from_intent`. The downstream
indoor-temperature trajectories under the same weather realization
show how a single sentence change reshapes the schedule.
""")
    examples = st.multiselect("Pick examples",
        [
         "My elderly mother is staying, keep it warm no matter what.",
         "Guests coming over tonight, keep it warm but watch the bill.",
         "Try to keep it warm-ish if you can.",
         "Newborn in the house, must stay warm tonight.",
        ],
        default=[
         "My elderly mother is staying, keep it warm no matter what.",
         "Guests coming over tonight, keep it warm but watch the bill.",
         "Try to keep it warm-ish if you can.",
        ])

    if st.button("Run alpha(z) demo", type="primary"):
        ctx = _real_context(horizon, peak_kwp)
        rows = []
        traj = []
        parser = SimulatedLLMParser()
        for u in examples:
            intent = parser.parse(u).to_dict()
            theta  = triangular_map(intent)
            alpha  = alpha_from_intent(intent)
            gw = None
            if intent.get("guest_flag") == 1 or intent.get("medical_context") == 1:
                ws = intent.get("window_start") or 19
                we = intent.get("window_end") or 23
                gw = (int(ws), int(we))
            scens = generate_scenarios(ctx, N_s=6, seed=42)
            sol = solve_stochastic(theta, scens, alpha=alpha,
                                   guest_window=gw,
                                   building=_building(E_bat, P_bat),
                                   time_limit_s=15)
            rows.append({
                "utterance": u,
                "comfort_intensity": intent.get("comfort_intensity"),
                "medical_context":   intent.get("medical_context"),
                "alpha": round(alpha, 3),
                "T_min": theta["T_min"],
                "feasible": sol["feasible"],
                "mean_CV (h)": (float(np.mean([
                    comfort_violation_count(sol["T_in"][w], theta["T_min"])
                    for w in range(sol["N_scenarios"])])) if sol["feasible"]
                    else None),
            })
            if sol.get("feasible"):
                T = sol["T_in"].mean(axis=0)
                traj.extend([{"hour": t, "T_in": T[t],
                              "utterance": u[:40] + "..."} for t in range(len(T))])

        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)
        if traj:
            dft = pd.DataFrame(traj)
            fig = px.line(dft, x="hour", y="T_in", color="utterance",
                          labels={"hour": "Hour (h)",
                                  "T_in": "Indoor temperature T_in (\u00b0C)",
                                  "utterance": "Utterance"},
                          title="Indoor temperature under varying linguistic intensity",
                    color_discrete_sequence=PRINT_PALETTE)
            style_for_print(fig)
            st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# Router
# =====================================================================
ROUTES = {
    "1. Overview":              tab_overview,
    "2. Single Command":        tab_single,
    "3. Linguistic Benchmark":  tab_benchmark,
    "4. Optimization Study":    tab_baselines,
    "5. Real Milan Data":       tab_realdata,
    "6. Sensitivity Analysis":  tab_sensitivity,
    "7. alpha(z) Mapping":      tab_novelty,
}
ROUTES[tab]()
