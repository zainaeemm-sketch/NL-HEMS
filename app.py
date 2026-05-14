"""
app.py — NL-HEMS-DR Bench dashboard
====================================
Supports two solver modes:
  • Conference (deterministic): the original CP-SAT solver in src/core.py.
    Reproduces the published conference-paper results exactly.
  • Journal (stochastic + PV/battery): src/core_stochastic.py with 2R2C
    thermal dynamics, PV-battery coupling, scenario-based SAA, and chance-
    constrained comfort. Adds CVaR, SCR, and Robust Feasibility Rate metrics.

The journal mode is auto-detected; if any of the new modules is missing,
the corresponding sidebar toggle is disabled and the rest of the app
continues to work.
"""

import os
import sys
import json
import inspect
import hashlib
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import altair as alt

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# --- Conference (deterministic) imports — required ---
from comparison_runner import load_jsonl, run_one_config, build_summary
from llm_parser import parse_command_by_mode
from clarification import should_clarify
from core import map_fuzzy_preferences, optimize_schedule

# --- Journal (stochastic) imports — optional, graceful degradation ---
JOURNAL_MODE_AVAILABLE = True
JOURNAL_IMPORT_ERROR = None
try:
    from scenarios import generate_scenarios, ForecastErrorModel  # noqa
    from core_stochastic import (                                  # noqa
        optimize_schedule_stochastic, PVBattery, Thermal2R2C,
    )
    from metrics_extended import replay_first_stage                # noqa
except Exception as _e:
    JOURNAL_MODE_AVAILABLE = False
    JOURNAL_IMPORT_ERROR = repr(_e)


# -----------------------------
# Page setup + styling
# -----------------------------
st.set_page_config(page_title="NL-HEMS-DR Dashboard", page_icon="⚡", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
.card { border: 1px solid rgba(128,128,128,0.18); border-radius: 14px; padding: 0.85rem 1rem; }
.small { color: #6b7280; font-size: 0.85rem; }
.big { font-size: 1.35rem; font-weight: 750; }
.section { font-size: 1.05rem; font-weight: 750; margin-top: 0.2rem; margin-bottom: 0.6rem; }
.journal-badge { background: #ffd966; color: #7f6000; padding: 2px 8px;
                  border-radius: 6px; font-size: 0.75rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ NL-HEMS-DR Benchmark Dashboard")
st.caption("Natural language → intent/fuzzy mapping → OR-Tools optimization → metrics → exports")


# -----------------------------
# Secrets → env vars (robust)
# -----------------------------
def apply_secrets_to_env():
    keys = ["OPENAI_BASE_URL", "OPENAI_API_KEY", "LLM_MODEL", "LLM_BACKEND"]
    for k in keys:
        try:
            v = st.secrets[k]
            if v:
                os.environ[k] = str(v)
        except Exception:
            pass
    try:
        for section_name in st.secrets.keys():
            section = st.secrets[section_name]
            if isinstance(section, dict):
                for k in keys:
                    if k in section and section[k]:
                        os.environ[k] = str(section[k])
    except Exception:
        pass

apply_secrets_to_env()


# -----------------------------
# Output saving + downloads
# -----------------------------
OUTPUT_DIR_CANDIDATES = [
    ROOT / "data" / "outputs",
    Path("/tmp") / "nl-hems-outputs",
]

def _first_writable_dir() -> Path:
    for d in OUTPUT_DIR_CANDIDATES:
        try:
            d.mkdir(parents=True, exist_ok=True)
            test = d / ".write_test"
            test.write_text("ok", encoding="utf-8")
            test.unlink(missing_ok=True)
            return d
        except Exception:
            continue
    raise RuntimeError("No writable output directory found (tried data/outputs and /tmp).")


def save_outputs(results_df: pd.DataFrame, summary_df: pd.DataFrame | None, prefix: str):
    out_dir = _first_writable_dir()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_csv = out_dir / f"{prefix}_{ts}.csv"
    results_json = out_dir / f"{prefix}_{ts}.json"
    summary_csv = out_dir / f"{prefix}_summary_{ts}.csv"
    results_df.to_csv(results_csv, index=False)
    results_json.write_text(json.dumps(results_df.to_dict("records"), indent=2), encoding="utf-8")
    summary_path = None
    if summary_df is not None and not summary_df.empty:
        summary_df.to_csv(summary_csv, index=False)
        summary_path = summary_csv
    return out_dir, results_csv, summary_path, results_json


def download_buttons(results_df: pd.DataFrame, summary_df: pd.DataFrame | None, prefix: str):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "⬇️ Results CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{prefix}_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c2:
        st.download_button(
            "⬇️ Results JSON",
            data=json.dumps(results_df.to_dict("records"), indent=2).encode("utf-8"),
            file_name=f"{prefix}_results.json",
            mime="application/json",
            use_container_width=True
        )
    with c3:
        if summary_df is not None and not summary_df.empty:
            st.download_button(
                "⬇️ Summary CSV",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{prefix}_summary.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.download_button(
                "⬇️ Summary CSV",
                data=b"",
                file_name=f"{prefix}_summary.csv",
                mime="text/csv",
                disabled=True,
                use_container_width=True
            )


def list_output_files() -> pd.DataFrame:
    rows = []
    for d in OUTPUT_DIR_CANDIDATES:
        if d.exists():
            for f in sorted(d.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
                if f.name == ".write_test":
                    continue
                rows.append({
                    "dir": str(d),
                    "name": f.name,
                    "size": f.stat().st_size,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds"),
                    "path": str(f),
                })
    return pd.DataFrame(rows)


# -----------------------------
# Navigation
# -----------------------------
PAGES = ["Dashboard", "Playground", "Dataset Editor", "Files"]
if "page_stack" not in st.session_state:
    st.session_state.page_stack = ["Dashboard"]

with st.sidebar:
    st.header("Navigation")
    current_page = st.radio(
        "Go to",
        PAGES,
        index=PAGES.index(st.session_state.page_stack[-1]),
        label_visibility="collapsed"
    )
    if current_page != st.session_state.page_stack[-1]:
        st.session_state.page_stack.append(current_page)
    if st.button("⬅ Back", use_container_width=True):
        if len(st.session_state.page_stack) > 1:
            st.session_state.page_stack.pop()
            st.rerun()

page = st.session_state.page_stack[-1]


# -----------------------------
# Sidebar: run settings, solver mode, context overrides
# -----------------------------
with st.sidebar:
    st.markdown("---")
    st.header("Run Settings")

    parser_mode = st.selectbox("Parser mode", ["stub", "llm"], index=0)
    comfort_mode = st.selectbox("Comfort mode", ["hard_guest_comfort", "soft_comfort"], index=0)
    enforce_hard = (comfort_mode == "hard_guest_comfort")

    dataset_path = st.text_input(
        "Dataset (JSONL)",
        value=str(ROOT / "data" / "samples" / "benchmark_samples.jsonl")
    )

    # ------------------ Solver mode (NEW) ------------------
    st.markdown("---")
    st.subheader("Solver Mode")
    if not JOURNAL_MODE_AVAILABLE:
        st.info(
            "Journal mode unavailable — falling back to conference solver. "
            "Add `scenarios.py`, `core_stochastic.py`, and `metrics_extended.py` "
            "to `src/` to enable it."
        )
        with st.expander("Import error details"):
            st.code(JOURNAL_IMPORT_ERROR or "n/a")
        use_stochastic = False
    else:
        solver_mode = st.radio(
            "Choose solver",
            ["Conference (deterministic)", "Journal (stochastic + PV/battery)"],
            index=0,
            help="Conference reproduces the published paper. Journal adds 2R2C "
                 "thermal, PV, battery, scenario tree, and CVaR/SCR/RFR metrics.",
        )
        use_stochastic = (solver_mode.startswith("Journal"))

    # ------------------ Stochastic parameters (NEW) ------------------
    if use_stochastic:
        st.markdown("##### Uncertainty (scenarios)")
        n_scenarios = st.slider("Number of scenarios (N_s)", 1, 30, 10)
        sigma_T = st.slider("σ on outdoor temp (°C)", 0.0, 4.0, 1.5, 0.1)
        sigma_p = st.slider("σ on price (fraction)", 0.0, 0.5, 0.15, 0.01)
        alpha_cc = st.slider(
            "Comfort chance-constraint α", 0.0, 0.5, 0.2, 0.05,
            help="Probability of comfort violation allowed across scenarios. "
                 "α=0 ≡ hard. Note: with uniform π, α must ≥ 1/N_s to allow "
                 "even one violator (so e.g. N_s=5 forces α≥0.2).",
        )
        rho_noise = st.slider("AR(1) persistence of noise (ρ)", 0.0, 0.95, 0.6, 0.05)

        st.markdown("##### PV / Battery")
        pv_peak = st.slider("PV peak (kW)", 0.0, 8.0, 3.0, 0.5)
        bat_kwh = st.slider("Battery capacity (kWh)", 0.0, 30.0, 10.0, 1.0)
        bat_p_max = st.slider("Battery max power (kW)", 0.0, 10.0, 5.0, 0.5)
        soc_init = st.slider("Initial SoC (fraction)", 0.1, 0.95, 0.5, 0.05)
        feed_in_frac = st.slider(
            "Feed-in tariff (fraction of buy price)", 0.0, 1.0, 0.5, 0.05
        )

        st.markdown("##### 2R2C thermal")
        c_in_kwhK = st.slider("Air heat capacity C_in (kWh/K)", 0.5, 5.0, 1.5, 0.1)
        c_m_kwhK = st.slider("Mass heat capacity C_m (kWh/K)", 2.0, 20.0, 8.0, 0.5)
        kappa_kw = st.slider("HVAC thermal power κ (kW)", 1.0, 8.0, 3.0, 0.5)
    else:
        n_scenarios = 1
        sigma_T = sigma_p = 0.0
        alpha_cc = 0.0
        rho_noise = 0.0
        pv_peak = 0.0
        bat_kwh = bat_p_max = 0.0
        soc_init = 0.5
        feed_in_frac = 0.5
        c_in_kwhK = 1.5
        c_m_kwhK = 8.0
        kappa_kw = 3.0

    # ------------------ Context controls (unchanged) ------------------
    st.markdown("---")
    st.subheader("Context Controls")
    use_sample_context = st.checkbox("Use per-sample context (if present)", value=True)
    enable_overrides = st.checkbox("Override context for runs", value=True)

    h_start = st.slider("Horizon start hour", 0, 23, 16)
    h_end = st.slider("Horizon end hour", 1, 24, 24)

    initial_temp_c = st.slider("Initial temp (°C)", 15.0, 26.0, 20.0, 0.1)
    loss_c = st.slider("Heat loss (°C/hour) — conference solver", 0.0, 1.5, 0.4, 0.05)
    heat_c = st.slider("Heating gain (°C/hour ON) — conference solver", 0.0, 2.0, 0.8, 0.05)

    hours_list = list(range(h_start, h_end))
    dr_hours = st.multiselect("DR event hours", options=hours_list,
                               default=[h for h in [18, 19] if h in hours_list])

    st.caption("Price levels (edit to change schedules)")
    default_price = []
    for h in hours_list:
        val = 2
        if h == 17: val = 3
        if h in (18, 20): val = 8
        if h == 19: val = 9
        if h == 21: val = 5
        if h == 22: val = 3
        if h == 23: val = 2
        default_price.append(val)

    if "price_df" not in st.session_state or st.session_state.get("price_df_hours") != hours_list:
        st.session_state.price_df = pd.DataFrame({"hour": hours_list, "price": default_price})
        st.session_state.price_df_hours = hours_list

    price_df = st.data_editor(
        st.session_state.price_df,
        use_container_width=True,
        num_rows="fixed",
        key="price_editor"
    )
    st.session_state.price_df = price_df
    price_override = {int(r["hour"]): int(r["price"]) for _, r in price_df.iterrows()}

    st.caption("Penalty weight multipliers")
    comfort_scale = st.slider("Comfort weight ×", 0.2, 3.0, 1.0, 0.1)
    cost_scale = st.slider("Cost weight ×", 0.2, 3.0, 1.0, 0.1)
    dr_scale = st.slider("DR weight ×", 0.2, 3.0, 1.0, 0.1)

    override_context = None
    if enable_overrides:
        override_context = {
            "horizon_start_hour": h_start,
            "horizon_end_hour": h_end,
            "initial_temp_c": initial_temp_c,
            "loss_c_per_hour": loss_c,
            "heat_c_per_hour": heat_c,
            "dr_event_hours": dr_hours,
            "price": price_override,
            "comfort_weight_scale": comfort_scale,
            "cost_weight_scale": cost_scale,
            "dr_weight_scale": dr_scale,
        }

    st.markdown("---")
    st.subheader("LLM Status")
    st.write("OPENAI_BASE_URL:", os.getenv("OPENAI_BASE_URL", "(not set)"))
    st.write("LLM_MODEL:", os.getenv("LLM_MODEL", "(not set)"))
    st.write("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))

    st.markdown("---")
    run_selected = st.button("Run Selected Benchmark", use_container_width=True)
    run_full = st.button("Run Full Comparison (stub/llm × soft/hard)", use_container_width=True)


# -----------------------------
# Session state
# -----------------------------
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()
if "summary_df" not in st.session_state:
    st.session_state.summary_df = pd.DataFrame()
if "saved_info" not in st.session_state:
    st.session_state.saved_info = None
if "last_mode" not in st.session_state:
    st.session_state.last_mode = "conference"


# -----------------------------
# Conference solver runners (unchanged from original)
# -----------------------------
def run_one_config_compat(samples, parser_mode: str, comfort_mode: dict,
                          override_ctx: dict | None, use_sample_ctx: bool):
    sig = inspect.signature(run_one_config)
    kwargs = {}
    if "override_context" in sig.parameters:
        kwargs["override_context"] = override_ctx
    if "use_sample_context" in sig.parameters:
        kwargs["use_sample_context"] = use_sample_ctx
    return run_one_config(samples, parser_mode=parser_mode, comfort_mode=comfort_mode, **kwargs)


# -----------------------------
# Journal-mode helpers (NEW)
# -----------------------------
def _adapt_mapped_for_stochastic(mapped: dict) -> dict:
    """
    Translate map_fuzzy_preferences output keys into the canonical names
    expected by optimize_schedule_stochastic. Adds missing aliases without
    overwriting anything already present.
    """
    if not isinstance(mapped, dict):
        return mapped
    adapted = dict(mapped)
    key_aliases = {
        "target_temp_c": ["T_tar", "target_temp", "target", "tar_temp_c"],
        "min_temp_c": ["T_min", "min_temp", "t_min", "min_temp_C"],
        "comfort_weight": ["lam_comf", "lambda_comfort", "lambda_comf", "w_comfort"],
        "cost_weight": ["lam_cost", "lambda_cost", "w_cost"],
        "dr_weight": ["lam_dr", "lambda_dr", "w_dr"],
        "switching_weight": ["lam_sw", "lambda_sw", "w_sw", "switching_penalty"],
        "min_comfort_weight": ["lam_min", "lambda_min", "w_min"],
        "guest_window": ["window", "comfort_window", "guest_hours", "W"],
        "guest_event": ["is_guest", "has_guest", "g"],
    }
    for canonical, aliases in key_aliases.items():
        if canonical not in adapted:
            for alias in aliases:
                if alias in adapted:
                    adapted[canonical] = adapted[alias]
                    break
    # Sensible defaults if a key is still missing
    adapted.setdefault("target_temp_c", 21.0)
    adapted.setdefault("min_temp_c", 19.5)
    adapted.setdefault("comfort_weight", 5.0)
    adapted.setdefault("cost_weight", 1.0)
    adapted.setdefault("dr_weight", 3.0)
    adapted.setdefault("switching_weight", 0.1)
    adapted.setdefault("min_comfort_weight", 2.0 * float(adapted.get("comfort_weight", 5.0)))
    adapted.setdefault("guest_event", False)
    adapted.setdefault("guest_window", [])
    return adapted


def _seed_from(sample_id: str, offset: int = 0) -> int:
    """Stable per-sample seed for reproducible scenario generation."""
    h = hashlib.md5(str(sample_id).encode("utf-8")).hexdigest()[:8]
    return (int(h, 16) + offset) % (2**31)


def _merge_context(sample: dict, override_ctx: dict | None, use_sample_ctx: bool) -> dict:
    """Merge per-sample context with global overrides (overrides win)."""
    ctx = {}
    if use_sample_ctx:
        sc = sample.get("context", {})
        if isinstance(sc, dict):
            ctx.update(sc)
    if override_ctx:
        ctx.update(override_ctx)
    return ctx


def _build_pvb() -> "PVBattery":
    return PVBattery(
        pv_enabled=(pv_peak > 0),
        battery_enabled=(bat_kwh > 0),
        energy_kwh=float(bat_kwh),
        p_max_kw=float(bat_p_max),
        soc_init_frac=float(soc_init),
        price_sell_scale=float(feed_in_frac),
    )


def _build_thermal() -> "Thermal2R2C":
    return Thermal2R2C(
        C_in=float(c_in_kwhK),
        C_m=float(c_m_kwhK),
        kappa_kw=float(kappa_kw),
    )


def _build_err_model(seed: int) -> "ForecastErrorModel":
    return ForecastErrorModel(
        sigma_temp_c=float(sigma_T),
        sigma_price_frac=float(sigma_p),
        rho=float(rho_noise),
        seed=int(seed),
    )


def run_one_sample_stochastic(sample: dict, parser_mode_: str, cmode: dict) -> dict:
    """Run one benchmark sample through the stochastic pipeline."""
    sid = sample.get("sample_id", "?")
    command = sample.get("command", "")
    gold = sample.get("gold", {}) or {}
    enforce_hard_ = bool(cmode.get("enforce_guest_hard_comfort", False))
    ctx = _merge_context(sample, override_context, use_sample_context)

    row_base = {
        "sample_id": sid,
        "parser_mode_requested": parser_mode_,
        "mode": cmode.get("name", ""),
        "command": command,
    }
    try:
        intent, meta = parse_command_by_mode(command, parser_mode_)
        clar, q = should_clarify(command, intent)
        intent.clarification_needed = clar
        mapped_raw = map_fuzzy_preferences(intent, enforce_guest_hard_comfort=enforce_hard_)
        mapped = _adapt_mapped_for_stochastic(mapped_raw)
    except Exception as e:
        return {**row_base, "status": "parse_error", "error": str(e)}

    try:
        train_scs = generate_scenarios(
            ctx, n_scenarios=int(n_scenarios),
            error_model=_build_err_model(seed=_seed_from(sid, 0)),
            pv_peak_kw=float(pv_peak),
        )
        test_scs = generate_scenarios(
            ctx, n_scenarios=20,
            error_model=_build_err_model(seed=_seed_from(sid, 9999)),
            pv_peak_kw=float(pv_peak),
        )
        pvb = _build_pvb()
        result = optimize_schedule_stochastic(
            mapped, ctx, train_scs,
            pv_battery_params=pvb,
            thermal_params=_build_thermal(),
            alpha_comfort=float(alpha_cc),
            enforce_guest_hard_comfort=enforce_hard_,
        )
        if result.get("status") != "ok":
            return {
                **row_base,
                "status": result.get("status", "infeasible"),
                "diagnostics": json.dumps(result.get("diagnostics", {})),
                "hints": " | ".join(result.get("hints", [])),
            }

        rfr_info = replay_first_stage(
            result["schedule"], ctx, test_scs, mapped, pv_battery_params=pvb,
        )
        metrics = result.get("metrics", {}) or {}
        gold_guest = bool(gold.get("expect_guest_event", False))
        pred_guest = bool(mapped.get("guest_event", False))

        return {
            **row_base,
            "status": "ok",
            "n_scenarios": metrics.get("n_scenarios", int(n_scenarios)),
            "comfort_violation_minutes": metrics.get("expected_comfort_violation_minutes"),
            "cvar_comfort_minutes": metrics.get("cvar_comfort_violation_minutes"),
            "dr_compliance_score": metrics.get("dr_compliance_score"),
            "self_consumption_ratio": metrics.get("self_consumption_ratio"),
            "expected_cost": metrics.get("expected_cost"),
            "hvac_on_hours": metrics.get("hvac_on_hours"),
            "avg_temp": metrics.get("avg_temp_across_scenarios"),
            "objective_value": metrics.get("objective_value"),
            "rfr": rfr_info.get("rfr"),
            "cvar_oos_minutes": rfr_info.get("cvar_comfort_oos"),
            "expected_cv_oos_minutes": rfr_info.get("expected_cv_oos"),
            "parser_used": (meta or {}).get("parser_used"),
            "parser_fallback": (meta or {}).get("fallback"),
            "guest_event_pred": pred_guest,
            "guest_event_gold": gold_guest,
            "guest_acc": 1 if pred_guest == gold_guest else 0,
        }
    except Exception as e:
        return {**row_base, "status": "solve_error", "error": str(e)}


def run_one_config_stochastic(samples, parser_mode_: str, comfort_mode_: dict) -> list[dict]:
    return [run_one_sample_stochastic(s, parser_mode_, comfort_mode_) for s in samples]


def build_summary_stochastic(df: pd.DataFrame) -> pd.DataFrame:
    """Simple aggregation for stochastic results."""
    if df is None or df.empty:
        return pd.DataFrame()
    ok = df[df["status"].astype(str) == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    num_cols = [
        "comfort_violation_minutes", "cvar_comfort_minutes",
        "dr_compliance_score", "self_consumption_ratio",
        "expected_cost", "hvac_on_hours", "avg_temp",
        "objective_value", "rfr", "cvar_oos_minutes", "guest_acc",
    ]
    for c in num_cols:
        if c in ok.columns:
            ok[c] = pd.to_numeric(ok[c], errors="coerce")
    grp = ok.groupby(["parser_mode_requested", "mode"], dropna=False)
    out = grp.agg({c: "mean" for c in num_cols if c in ok.columns}).reset_index()
    out.insert(0, "n_samples", grp.size().values)
    return out


# -----------------------------
# Run actions (work from any page)
# -----------------------------
def run_selected_benchmark():
    samples = load_jsonl(Path(dataset_path))
    cmode = {"name": "hard_guest_comfort" if enforce_hard else "soft_comfort",
             "enforce_guest_hard_comfort": enforce_hard}

    if use_stochastic and JOURNAL_MODE_AVAILABLE:
        rows = run_one_config_stochastic(samples, parser_mode_=parser_mode, comfort_mode_=cmode)
        df = pd.DataFrame(rows)
        summary = build_summary_stochastic(df)
        prefix = f"benchmark_journal_{parser_mode}_{cmode['name']}"
        st.session_state.last_mode = "journal"
    else:
        rows = run_one_config_compat(
            samples, parser_mode=parser_mode, comfort_mode=cmode,
            override_ctx=override_context, use_sample_ctx=use_sample_context,
        )
        df = pd.DataFrame(rows)
        summary = build_summary(df)
        prefix = f"benchmark_{parser_mode}_{cmode['name']}"
        st.session_state.last_mode = "conference"

    st.session_state.results_df = df
    st.session_state.summary_df = summary
    out_dir, r_csv, s_csv, r_json = save_outputs(df, summary, prefix=prefix)
    st.session_state.saved_info = {
        "out_dir": str(out_dir),
        "results_csv": str(r_csv),
        "summary_csv": str(s_csv) if s_csv else None,
        "results_json": str(r_json),
    }


def run_full_comparison():
    samples = load_jsonl(Path(dataset_path))
    configs = [("stub", False), ("stub", True), ("llm", False), ("llm", True)]
    all_rows = []

    if use_stochastic and JOURNAL_MODE_AVAILABLE:
        for pmode, hard in configs:
            cmode = {"name": "hard_guest_comfort" if hard else "soft_comfort",
                     "enforce_guest_hard_comfort": hard}
            all_rows.extend(run_one_config_stochastic(samples, parser_mode_=pmode, comfort_mode_=cmode))
        df = pd.DataFrame(all_rows)
        summary = build_summary_stochastic(df)
        prefix = "comparison_journal_all"
        st.session_state.last_mode = "journal"
    else:
        for pmode, hard in configs:
            cmode = {"name": "hard_guest_comfort" if hard else "soft_comfort",
                     "enforce_guest_hard_comfort": hard}
            all_rows.extend(
                run_one_config_compat(samples, parser_mode=pmode, comfort_mode=cmode,
                                       override_ctx=override_context, use_sample_ctx=use_sample_context)
            )
        df = pd.DataFrame(all_rows)
        summary = build_summary(df)
        prefix = "comparison_all"
        st.session_state.last_mode = "conference"

    st.session_state.results_df = df
    st.session_state.summary_df = summary
    out_dir, r_csv, s_csv, r_json = save_outputs(df, summary, prefix=prefix)
    st.session_state.saved_info = {
        "out_dir": str(out_dir),
        "results_csv": str(r_csv),
        "summary_csv": str(s_csv) if s_csv else None,
        "results_json": str(r_json),
    }


if run_selected:
    try:
        with st.spinner("Running benchmark..."):
            run_selected_benchmark()
        st.success("Benchmark run complete. Outputs saved.")
        st.session_state.page_stack = ["Dashboard"]
        st.rerun()
    except Exception as e:
        st.exception(e)

if run_full:
    try:
        with st.spinner("Running full comparison..."):
            run_full_comparison()
        st.success("Comparison run complete. Outputs saved.")
        st.session_state.page_stack = ["Dashboard"]
        st.rerun()
    except Exception as e:
        st.exception(e)


# -----------------------------
# Charts
# -----------------------------
def kpi_cards_conference(df: pd.DataFrame):
    ok = df.copy()
    if "status" in ok.columns:
        ok["status"] = ok["status"].astype(str).str.strip().str.lower()
        ok = ok[ok["status"] == "ok"].copy()

    feasibility = (len(ok) / len(df)) if len(df) else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='card'><div class='small'>Rows</div><div class='big'>{len(df)}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'><div class='small'>Feasibility</div><div class='big'>{feasibility:.2f}</div></div>", unsafe_allow_html=True)

    if len(ok) and "comfort_violation_minutes" in ok.columns:
        ok["comfort_violation_minutes"] = pd.to_numeric(ok["comfort_violation_minutes"], errors="coerce")
        val = ok["comfort_violation_minutes"].mean()
        c3.markdown(
            f"<div class='card'><div class='small'>Avg comfort violation (min)</div><div class='big'>{val:.1f}</div></div>"
            if pd.notna(val) else
            "<div class='card'><div class='small'>Avg comfort violation (min)</div><div class='big'>n/a</div></div>",
            unsafe_allow_html=True)
    else:
        c3.markdown("<div class='card'><div class='small'>Avg comfort violation (min)</div><div class='big'>n/a</div></div>", unsafe_allow_html=True)

    if len(ok) and "dr_compliance_score" in ok.columns:
        ok["dr_compliance_score"] = pd.to_numeric(ok["dr_compliance_score"], errors="coerce")
        val = ok["dr_compliance_score"].mean()
        c4.markdown(
            f"<div class='card'><div class='small'>Avg DR compliance</div><div class='big'>{val:.2f}</div></div>"
            if pd.notna(val) else
            "<div class='card'><div class='small'>Avg DR compliance</div><div class='big'>n/a</div></div>",
            unsafe_allow_html=True)
    else:
        c4.markdown("<div class='card'><div class='small'>Avg DR compliance</div><div class='big'>n/a</div></div>", unsafe_allow_html=True)


def kpi_cards_journal(df: pd.DataFrame):
    """Extra KPI cards for stochastic mode."""
    ok = df[df["status"].astype(str) == "ok"].copy() if "status" in df.columns else df.copy()

    def _safe_mean(col):
        if col in ok.columns:
            return pd.to_numeric(ok[col], errors="coerce").mean()
        return float("nan")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f"<div class='card'><div class='small'>E[Comfort Violation] (min) "
        f"<span class='journal-badge'>journal</span></div>"
        f"<div class='big'>{_safe_mean('comfort_violation_minutes'):.1f}</div></div>",
        unsafe_allow_html=True)
    c2.markdown(
        f"<div class='card'><div class='small'>CVaR₀.₂ Comfort (min) "
        f"<span class='journal-badge'>journal</span></div>"
        f"<div class='big'>{_safe_mean('cvar_comfort_minutes'):.1f}</div></div>",
        unsafe_allow_html=True)
    c3.markdown(
        f"<div class='card'><div class='small'>Robust Feasibility Rate "
        f"<span class='journal-badge'>journal</span></div>"
        f"<div class='big'>{_safe_mean('rfr'):.2f}</div></div>",
        unsafe_allow_html=True)
    scr_val = _safe_mean('self_consumption_ratio')
    c4.markdown(
        f"<div class='card'><div class='small'>Self-Consumption Ratio "
        f"<span class='journal-badge'>journal</span></div>"
        f"<div class='big'>{scr_val:.2f}</div></div>"
        if pd.notna(scr_val) else
        "<div class='card'><div class='small'>Self-Consumption Ratio "
        "<span class='journal-badge'>journal</span></div>"
        "<div class='big'>n/a</div></div>",
        unsafe_allow_html=True)


def charts(df: pd.DataFrame):
    plot_df = df.copy()
    if "status" in plot_df.columns:
        plot_df["status"] = plot_df["status"].astype(str).str.strip().str.lower()
        plot_df = plot_df[plot_df["status"] == "ok"].copy()

    if plot_df.empty:
        st.warning("No rows with status='ok' to plot. Check the results table for errors.")
        return

    if "mode" not in plot_df.columns:
        plot_df["mode"] = "run"
    if "parser_mode_requested" not in plot_df.columns:
        plot_df["parser_mode_requested"] = "unknown"
    if "sample_id" not in plot_df.columns:
        plot_df["sample_id"] = plot_df.index.astype(str)

    for col in ["comfort_violation_minutes", "dr_compliance_score", "hvac_on_hours",
                "cvar_comfort_minutes", "self_consumption_ratio", "rfr", "expected_cost"]:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    # Per-sample bars
    st.markdown("<div class='section'>Per-sample charts</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if "comfort_violation_minutes" in plot_df.columns and plot_df["comfort_violation_minutes"].notna().any():
            d = plot_df.dropna(subset=["comfort_violation_minutes"])
            ch = alt.Chart(d).mark_bar().encode(
                x=alt.X("sample_id:N", title="Sample"),
                y=alt.Y("comfort_violation_minutes:Q", title="Comfort violation (min)"),
                color=alt.Color("mode:N", title="Mode"),
                tooltip=["sample_id", "parser_mode_requested", "mode",
                         "comfort_violation_minutes", "dr_compliance_score", "hvac_on_hours"],
            ).properties(height=320)
            st.altair_chart(ch, use_container_width=True)
        else:
            st.info("No numeric comfort_violation_minutes values to plot.")
    with c2:
        if "dr_compliance_score" in plot_df.columns and plot_df["dr_compliance_score"].notna().any():
            d = plot_df.dropna(subset=["dr_compliance_score"])
            ch = alt.Chart(d).mark_bar().encode(
                x=alt.X("sample_id:N", title="Sample"),
                y=alt.Y("dr_compliance_score:Q", title="DR compliance", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("mode:N", title="Mode"),
                tooltip=["sample_id", "parser_mode_requested", "mode",
                         "dr_compliance_score", "comfort_violation_minutes"],
            ).properties(height=320)
            st.altair_chart(ch, use_container_width=True)
        else:
            st.info("No numeric dr_compliance_score values to plot.")

    # Journal-mode extras
    if st.session_state.last_mode == "journal" and "cvar_comfort_minutes" in plot_df.columns:
        st.markdown("<div class='section'>Stochastic-mode charts</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            d = plot_df.dropna(subset=["cvar_comfort_minutes"])
            if not d.empty:
                ch = alt.Chart(d).mark_bar().encode(
                    x=alt.X("sample_id:N", title="Sample"),
                    y=alt.Y("cvar_comfort_minutes:Q", title="CVaR₀.₂ comfort (min)"),
                    color=alt.Color("mode:N"),
                    tooltip=["sample_id", "parser_mode_requested", "mode",
                             "comfort_violation_minutes", "cvar_comfort_minutes"],
                ).properties(height=320)
                st.altair_chart(ch, use_container_width=True)
        with c2:
            if "rfr" in plot_df.columns and plot_df["rfr"].notna().any():
                d = plot_df.dropna(subset=["rfr"])
                ch = alt.Chart(d).mark_bar().encode(
                    x=alt.X("sample_id:N", title="Sample"),
                    y=alt.Y("rfr:Q", title="Robust Feasibility Rate", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("mode:N"),
                    tooltip=["sample_id", "rfr", "cvar_oos_minutes", "expected_cv_oos_minutes"],
                ).properties(height=320)
                st.altair_chart(ch, use_container_width=True)

        if "self_consumption_ratio" in plot_df.columns and "expected_cost" in plot_df.columns:
            d = plot_df.dropna(subset=["self_consumption_ratio", "expected_cost"])
            if not d.empty:
                st.markdown("<div class='section'>SCR vs Expected Cost</div>", unsafe_allow_html=True)
                ch = alt.Chart(d).mark_circle(size=140, opacity=0.85).encode(
                    x=alt.X("expected_cost:Q", title="Expected cost"),
                    y=alt.Y("self_consumption_ratio:Q", title="Self-consumption ratio",
                             scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("mode:N"),
                    shape=alt.Shape("parser_mode_requested:N"),
                    tooltip=["sample_id", "mode", "parser_mode_requested",
                             "expected_cost", "self_consumption_ratio", "dr_compliance_score"],
                ).properties(height=360).interactive()
                st.altair_chart(ch, use_container_width=True)

    # Tradeoff scatter (kept from original)
    st.markdown("<div class='section'>Tradeoff (Comfort vs DR)</div>", unsafe_allow_html=True)
    needed = {"comfort_violation_minutes", "dr_compliance_score"}
    if not needed.issubset(plot_df.columns):
        st.info(f"Missing columns for tradeoff plot: {needed - set(plot_df.columns)}")
        return

    base = plot_df.dropna(subset=["comfort_violation_minutes", "dr_compliance_score"]).copy()
    if base.empty:
        st.warning("Tradeoff plot has no points to draw.")
        return

    def jitter_from_id(s: str):
        h = int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)
        return ((h % 9) - 4) * 0.9, (((h // 9) % 9) - 4) * 0.01

    jx_list, jy_list = [], []
    for sid in base["sample_id"].astype(str).tolist():
        jx, jy = jitter_from_id(sid)
        jx_list.append(jx); jy_list.append(jy)
    base["x_j"] = base["comfort_violation_minutes"] + pd.Series(jx_list)
    base["y_j"] = (base["dr_compliance_score"] + pd.Series(jy_list)).clip(0, 1)

    pts = alt.Chart(base).mark_circle(size=140, opacity=0.85).encode(
        x=alt.X("x_j:Q", title="Comfort violation (min)"),
        y=alt.Y("y_j:Q", title="DR compliance", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("mode:N", title="Mode"),
        shape=alt.Shape("parser_mode_requested:N", title="Parser"),
        tooltip=["sample_id", "parser_mode_requested", "mode",
                 "comfort_violation_minutes", "dr_compliance_score", "hvac_on_hours"],
    )
    labels = alt.Chart(base).mark_text(align="left", dx=6, dy=-6, fontSize=11).encode(
        x="x_j:Q", y="y_j:Q", text="sample_id:N",
    )
    st.altair_chart((pts + labels).properties(height=380).interactive(), use_container_width=True)


# -----------------------------
# Pages
# -----------------------------
if page == "Dashboard":
    df = st.session_state.results_df
    summary = st.session_state.summary_df

    if df is None or df.empty:
        st.info("Click **Run Selected Benchmark** in the sidebar to generate results and outputs.")
    else:
        mode_label = "Journal (stochastic + PV/battery)" if st.session_state.last_mode == "journal" else "Conference (deterministic)"
        st.caption(f"Last run: **{mode_label}**")

        st.markdown("<div class='section'>Downloads</div>", unsafe_allow_html=True)
        download_buttons(df, summary, prefix="nl_hems")
        if st.session_state.saved_info:
            st.caption(f"Saved on server in: {st.session_state.saved_info['out_dir']}")
            st.caption(f"CSV: {st.session_state.saved_info['results_csv']}")

        kpi_cards_conference(df)
        if st.session_state.last_mode == "journal":
            kpi_cards_journal(df)

        charts(df)

        st.markdown("<div class='section'>Tables</div>", unsafe_allow_html=True)
        if summary is not None and not summary.empty:
            st.write("Summary")
            st.dataframe(summary, use_container_width=True)
        st.write("Detailed results")
        st.dataframe(df, use_container_width=True)


elif page == "Files":
    st.markdown("<div class='section'>Outputs</div>", unsafe_allow_html=True)
    st.write("Output directories checked:")
    for d in OUTPUT_DIR_CANDIDATES:
        st.write("-", str(d), "(exists)" if d.exists() else "(missing)")

    files_df = list_output_files()
    if files_df.empty:
        st.warning("No output files yet. Run a benchmark from the sidebar first.")
        if st.button("Create test output file", use_container_width=True):
            try:
                d = _first_writable_dir()
                tf = d / f"test_{datetime.now().strftime('%H%M%S')}.txt"
                tf.write_text("test ok", encoding="utf-8")
                st.success(f"Created {tf}")
                st.rerun()
            except Exception as e:
                st.exception(e)
    else:
        show_only_outputs = st.checkbox("Show only benchmark output files (csv/json/txt)", value=True)
        view_df = files_df.copy()
        if show_only_outputs:
            view_df = view_df[view_df["name"].str.lower().str.endswith((".csv", ".json", ".txt"))].copy()
        st.dataframe(view_df, use_container_width=True)

        pick = st.selectbox("Preview", view_df["path"].tolist())
        fp = Path(pick)
        if fp.suffix.lower() == ".csv":
            st.dataframe(pd.read_csv(fp).head(300), use_container_width=True)
        elif fp.suffix.lower() == ".json":
            content = json.loads(fp.read_text(encoding="utf-8"))
            st.json(content[:10] if isinstance(content, list) else content)
        else:
            st.write(fp.read_text(encoding="utf-8", errors="replace")[:3000])

        st.markdown("---")
        st.markdown("<div class='section'>Delete outputs</div>", unsafe_allow_html=True)
        st.caption("Deletes files from the app's container storage. On Streamlit Cloud, files may also disappear on restart.")
        candidates = view_df["path"].tolist()
        selected = st.multiselect("Select files to delete", candidates, default=[])
        colA, colB = st.columns(2)
        with colA:
            confirm = st.checkbox("I understand this will permanently delete selected files.", value=False)
            if st.button("🗑️ Delete selected", use_container_width=True, disabled=(not selected or not confirm)):
                deleted, errors = 0, []
                for p in selected:
                    try:
                        Path(p).unlink(missing_ok=True); deleted += 1
                    except Exception as e:
                        errors.append((p, str(e)))
                if deleted: st.success(f"Deleted {deleted} file(s).")
                if errors: st.error("Some files could not be deleted:"); st.write(errors[:10])
                st.rerun()
        with colB:
            confirm_all = st.checkbox("Confirm delete ALL listed files", value=False)
            if st.button("🔥 Delete ALL listed outputs", use_container_width=True, disabled=(not confirm_all)):
                deleted, errors = 0, []
                for p in candidates:
                    try:
                        Path(p).unlink(missing_ok=True); deleted += 1
                    except Exception as e:
                        errors.append((p, str(e)))
                if deleted: st.success(f"Deleted {deleted} file(s).")
                if errors: st.error("Some files could not be deleted:"); st.write(errors[:10])
                st.rerun()


elif page == "Dataset Editor":
    st.markdown("<div class='section'>Dataset editor (JSONL)</div>", unsafe_allow_html=True)
    p = Path(dataset_path)
    if not p.exists():
        st.error(f"File not found: {p}")
    else:
        txt = p.read_text(encoding="utf-8", errors="replace")
        edited = st.text_area("Edit JSONL", value=txt, height=450)
        c1, c2 = st.columns(2)
        if c1.button("Save dataset", use_container_width=True):
            p.write_text(edited, encoding="utf-8")
            st.success("Saved.")
        if c2.button("Validate JSONL", use_container_width=True):
            ok, bad = 0, []
            for i, line in enumerate(edited.splitlines(), start=1):
                s = line.strip()
                if not s: continue
                try:
                    json.loads(s); ok += 1
                except Exception as e:
                    bad.append((i, str(e), s[:160]))
            if bad:
                st.error(f"Invalid lines: {len(bad)}"); st.write(bad[:10])
            else:
                st.success(f"All good. Parsed {ok} JSON lines.")
        st.info('Tip: Add per-sample context like {"context":{"initial_temp_c":21,"dr_event_hours":[19,20],"price":{"16":2,...}}}')


elif page == "Playground":
    if use_stochastic and JOURNAL_MODE_AVAILABLE:
        st.markdown(
            "<div class='section'>Single-command playground "
            "<span class='journal-badge'>journal mode</span></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("<div class='section'>Single-command playground</div>", unsafe_allow_html=True)

    command = st.text_area(
        "Command",
        value="I have guests coming over tonight, keep the house warm but don't run up the bill.",
        height=120
    )
    colA, colB, colC = st.columns(3)
    play_parser = colA.selectbox("Parser", ["stub", "llm"], index=0)
    play_hard = colB.checkbox("Hard guest comfort", value=True)
    run_one = colC.button("Run", use_container_width=True)

    if run_one:
        try:
            intent, meta = parse_command_by_mode(command, play_parser)
            clar, q = should_clarify(command, intent)
            intent.clarification_needed = clar

            mapped_raw = map_fuzzy_preferences(intent, enforce_guest_hard_comfort=play_hard)
            ctx = override_context if enable_overrides else {}

            st.write({"parser_used": (meta or {}).get("parser_used"),
                       "fallback": (meta or {}).get("fallback"),
                       "error": (meta or {}).get("error")})
            st.write({"clarification_pred": clar, "clarification_question": q})

            with st.expander("Parsed intent (JSON)"):
                try:
                    st.json(intent.model_dump())
                except Exception:
                    st.write(intent)
            with st.expander("Fuzzy-mapped parameters"):
                st.json(mapped_raw)

            if use_stochastic and JOURNAL_MODE_AVAILABLE:
                mapped = _adapt_mapped_for_stochastic(mapped_raw)
                pvb = _build_pvb()
                with st.spinner(f"Running stochastic SAA with N_s={n_scenarios}..."):
                    scs = generate_scenarios(
                        ctx or {}, n_scenarios=int(n_scenarios),
                        error_model=_build_err_model(seed=42),
                        pv_peak_kw=float(pv_peak),
                    )
                    sched = optimize_schedule_stochastic(
                        mapped, ctx, scs,
                        pv_battery_params=pvb, thermal_params=_build_thermal(),
                        alpha_comfort=float(alpha_cc),
                        enforce_guest_hard_comfort=play_hard,
                    )

                if sched.get("status") != "ok":
                    st.error(f"Solver returned: {sched.get('status')}")
                    diag = sched.get("diagnostics")
                    if diag:
                        st.write("Diagnostics:")
                        st.json(diag)
                    hints = sched.get("hints", [])
                    if hints:
                        st.warning("Hints:\n" + "\n".join(f"• {h}" for h in hints))
                    st.stop()

                # ---- KPI strip ----
                m = sched.get("metrics", {}) or {}
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("E[CV] (min)", f"{m.get('expected_comfort_violation_minutes', 0):.1f}")
                c2.metric("CVaR₀.₂ CV (min)", f"{m.get('cvar_comfort_violation_minutes', 0):.1f}")
                c3.metric("DR compliance", f"{m.get('dr_compliance_score', 0):.2f}")
                scr = m.get('self_consumption_ratio')
                c4.metric("SCR", f"{scr:.2f}" if scr is not None else "n/a")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("E[cost]", f"{m.get('expected_cost', 0):.2f}")
                c6.metric("HVAC on-hours", f"{m.get('hvac_on_hours', 0)}")
                c7.metric("Avg T_in (°C)", f"{m.get('avg_temp_across_scenarios', 0):.2f}")
                c8.metric("Objective", f"{m.get('objective_value', 0):.2f}")

                # ---- First-stage schedule ----
                st.markdown("##### First-stage decisions (shared across scenarios)")
                fs_df = pd.DataFrame(sched["schedule"])
                fs_chart_y = alt.Chart(fs_df).mark_bar().encode(
                    x=alt.X("hour:Q", title="Hour"),
                    y=alt.Y("hvac_on:Q", title="HVAC", scale=alt.Scale(domain=[0, 1])),
                    color=alt.condition("datum.dr_event == 1",
                                          alt.value("#cc4040"), alt.value("#3b8e3b")),
                    tooltip=["hour", "hvac_on", "battery_mode", "dr_event", "guest_window"],
                ).properties(height=180)
                st.altair_chart(fs_chart_y, use_container_width=True)
                st.dataframe(fs_df, use_container_width=True)

                # ---- Per-scenario trajectories ----
                traces = sched.get("scenario_traces", []) or []
                if traces:
                    traces_df = pd.concat([
                        pd.DataFrame(tr["rows"]).assign(scenario=int(tr["scenario_id"]))
                        for tr in traces
                    ], ignore_index=True)

                    st.markdown("##### Indoor temperature trajectories across scenarios")
                    tin_chart = alt.Chart(traces_df).mark_line(opacity=0.55).encode(
                        x=alt.X("hour:Q", title="Hour"),
                        y=alt.Y("T_in:Q", title="T_in (°C)"),
                        color=alt.Color("scenario:N", legend=None),
                        tooltip=["hour", "scenario", "T_in", "T_m", "SoC_kwh", "P_grid_kw"],
                    ).properties(height=260)
                    tin_mean = alt.Chart(traces_df).mark_line(
                        color="#000000", strokeWidth=2.5
                    ).encode(
                        x="hour:Q",
                        y=alt.Y("mean(T_in):Q", title="T_in (°C)"),
                    )
                    st.altair_chart((tin_chart + tin_mean), use_container_width=True)

                    cA, cB = st.columns(2)
                    with cA:
                        st.markdown("##### Battery SoC")
                        soc_chart = alt.Chart(traces_df).mark_line(opacity=0.55).encode(
                            x="hour:Q",
                            y=alt.Y("SoC_kwh:Q", title="SoC (kWh)"),
                            color=alt.Color("scenario:N", legend=None),
                            tooltip=["hour", "scenario", "SoC_kwh", "P_ch_kw", "P_dis_kw"],
                        ).properties(height=220)
                        st.altair_chart(soc_chart, use_container_width=True)
                    with cB:
                        st.markdown("##### Grid power (+import / −export)")
                        grid_chart = alt.Chart(traces_df).mark_line(opacity=0.55).encode(
                            x="hour:Q",
                            y=alt.Y("P_grid_kw:Q", title="P_grid (kW)"),
                            color=alt.Color("scenario:N", legend=None),
                            tooltip=["hour", "scenario", "P_grid_kw", "PV_kw"],
                        ).properties(height=220)
                        st.altair_chart(grid_chart, use_container_width=True)

                    st.markdown("##### PV generation (per scenario)")
                    pv_chart = alt.Chart(traces_df).mark_line(opacity=0.55).encode(
                        x="hour:Q",
                        y=alt.Y("PV_kw:Q", title="PV (kW)"),
                        color=alt.Color("scenario:N", legend=None),
                    ).properties(height=180)
                    st.altair_chart(pv_chart, use_container_width=True)

                    with st.expander("Show full per-scenario trace table"):
                        st.dataframe(traces_df, use_container_width=True)

            else:
                # Conference path (unchanged)
                sched = optimize_schedule(mapped_raw, context=ctx)
                if sched.get("status") == "ok":
                    sdf = pd.DataFrame(sched["schedule"])
                    tchart = alt.Chart(sdf).mark_line(point=True).encode(
                        x=alt.X("hour:Q", title="Hour"),
                        y=alt.Y("temp_c:Q", title="Temp (°C)"),
                        tooltip=["hour", "temp_c", "hvac_on", "dr_event", "guest_window"]
                    ).properties(height=260)
                    hchart = alt.Chart(sdf).mark_bar().encode(
                        x=alt.X("hour:Q", title="Hour"),
                        y=alt.Y("hvac_on:Q", title="HVAC On (0/1)", scale=alt.Scale(domain=[0, 1])),
                        tooltip=["hour", "hvac_on", "price_level", "dr_event", "guest_window"]
                    ).properties(height=180)
                    st.altair_chart(tchart, use_container_width=True)
                    st.altair_chart(hchart, use_container_width=True)
                    st.dataframe(sdf, use_container_width=True)
                else:
                    st.json(sched)

        except Exception as e:
            st.exception(e)