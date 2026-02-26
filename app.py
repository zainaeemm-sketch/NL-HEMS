import os
import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import altair as alt

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from comparison_runner import load_jsonl, run_one_config, build_summary
from core import optimize_schedule, map_fuzzy_preferences
from llm_parser import parse_command_by_mode
from clarification import should_clarify


# -----------------------------
# Styling
# -----------------------------
st.set_page_config(page_title="NL-HEMS-DR Dashboard", page_icon="⚡", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
.card { border: 1px solid rgba(128,128,128,0.2); border-radius: 14px; padding: 0.9rem 1rem; }
.small { color: #6b7280; font-size: 0.85rem; }
.big { font-size: 1.4rem; font-weight: 750; }
.section { font-size: 1.05rem; font-weight: 750; margin-top: 0.2rem; margin-bottom: 0.6rem; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ NL-HEMS-DR Benchmark Dashboard")
st.caption("Natural-language → fuzzy preferences → optimized DR-aware schedule → benchmark metrics & figures")


# -----------------------------
# Secrets → env vars (robust)
# -----------------------------
def apply_secrets_to_env():
    keys = ["OPENAI_BASE_URL", "OPENAI_API_KEY", "LLM_MODEL", "LLM_BACKEND"]

    # root-level secrets
    for k in keys:
        try:
            v = st.secrets[k]
            if v:
                os.environ[k] = str(v)
        except Exception:
            pass

    # section-based secrets fallback
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
# Output helpers (Fix 1 + Fix 2)
# -----------------------------
def get_output_dir() -> Path:
    """
    Prefer writing inside the app folder, but fallback to /tmp if permissions are restricted.
    """
    primary = ROOT / "data" / "outputs"
    try:
        primary.mkdir(parents=True, exist_ok=True)
        testfile = primary / ".write_test"
        testfile.write_text("ok", encoding="utf-8")
        testfile.unlink(missing_ok=True)
        return primary
    except Exception:
        fallback = Path("/tmp") / "nl-hems-outputs"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def save_outputs(results_df: pd.DataFrame, summary_df: pd.DataFrame | None, prefix: str):
    """
    Always save output artifacts after a run.
    Returns paths for display in the app.
    """
    out_dir = get_output_dir()
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

    return results_csv, summary_path, results_json


def download_buttons(results_df: pd.DataFrame, summary_df: pd.DataFrame | None, prefix: str):
    """
    Fix 2: always allow downloading results (reliable even if server disk resets).
    """
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


# -----------------------------
# Navigation with Back button
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
# Sidebar controls (inputs + overrides)
# -----------------------------
with st.sidebar:
    st.markdown("---")
    st.header("Run Settings")

    parser_mode = st.selectbox("Parser mode", ["stub", "llm"], index=0)
    comfort_mode = st.selectbox("Comfort mode", ["hard_guest_comfort", "soft_comfort"], index=0)
    enforce_guest_hard_comfort = comfort_mode == "hard_guest_comfort"

    dataset_path = st.text_input(
        "Dataset (JSONL)",
        value=str(ROOT / "data" / "samples" / "benchmark_samples.jsonl")
    )

    st.markdown("---")
    st.subheader("Context Controls")
    use_sample_context = st.checkbox("Use per-sample context (if present)", value=True)
    enable_overrides = st.checkbox("Override context for runs", value=True)

    # Horizon
    h_start = st.slider("Horizon start hour", 0, 23, 16)
    h_end = st.slider("Horizon end hour", 1, 24, 24)

    # Thermal
    initial_temp_c = st.slider("Initial temp (°C)", 15.0, 26.0, 20.0, 0.1)
    loss_c = st.slider("Heat loss (°C/hour)", 0.0, 1.5, 0.4, 0.05)
    heat_c = st.slider("Heating gain (°C/hour when ON)", 0.0, 2.0, 0.8, 0.05)

    # DR hours
    hours_list = list(range(h_start, h_end))
    dr_hours = st.multiselect("DR event hours", options=hours_list, default=[h for h in [18, 19] if h in hours_list])

    # Price table
    st.caption("Price levels (edit to see different schedules)")
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

    # Weight multipliers
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
# Run helpers
# -----------------------------
def run_benchmark(parser_mode: str, enforce_hard: bool):
    samples = load_jsonl(Path(dataset_path))
    cmode = {"name": "hard_guest_comfort" if enforce_hard else "soft_comfort",
             "enforce_guest_hard_comfort": enforce_hard}
    rows = run_one_config(
        samples,
        parser_mode=parser_mode,
        comfort_mode=cmode,
        override_context=override_context,
        use_sample_context=use_sample_context,
    )
    df = pd.DataFrame(rows)
    summary = build_summary(df)
    return df, summary


def run_comparison_all():
    samples = load_jsonl(Path(dataset_path))
    configs = [("stub", False), ("stub", True), ("llm", False), ("llm", True)]
    all_rows = []
    prog = st.progress(0.0)
    for i, (pmode, hard) in enumerate(configs, start=1):
        cmode = {"name": "hard_guest_comfort" if hard else "soft_comfort",
                 "enforce_guest_hard_comfort": hard}
        all_rows.extend(
            run_one_config(
                samples,
                parser_mode=pmode,
                comfort_mode=cmode,
                override_context=override_context,
                use_sample_context=use_sample_context,
            )
        )
        prog.progress(i / len(configs))
    df = pd.DataFrame(all_rows)
    summary = build_summary(df)
    return df, summary


def kpi_cards(df: pd.DataFrame):
    ok = df[df["status"] == "ok"] if "status" in df.columns else pd.DataFrame()
    feasibility = len(ok) / len(df) if len(df) else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='card'><div class='small'>Rows</div><div class='big'>{len(df)}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'><div class='small'>Feasibility</div><div class='big'>{feasibility:.2f}</div></div>", unsafe_allow_html=True)
    if len(ok) and "comfort_violation_minutes" in ok.columns:
        c3.markdown(f"<div class='card'><div class='small'>Avg comfort violation (min)</div><div class='big'>{ok['comfort_violation_minutes'].mean():.1f}</div></div>", unsafe_allow_html=True)
    else:
        c3.markdown("<div class='card'><div class='small'>Avg comfort violation (min)</div><div class='big'>n/a</div></div>", unsafe_allow_html=True)
    if len(ok) and "dr_compliance_score" in ok.columns:
        c4.markdown(f"<div class='card'><div class='small'>Avg DR compliance</div><div class='big'>{ok['dr_compliance_score'].mean():.2f}</div></div>", unsafe_allow_html=True)
    else:
        c4.markdown("<div class='card'><div class='small'>Avg DR compliance</div><div class='big'>n/a</div></div>", unsafe_allow_html=True)


def charts(df: pd.DataFrame, summary: pd.DataFrame):
    ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()
    if ok.empty:
        st.warning("No successful rows to chart.")
        return

    st.markdown("<div class='section'>Per-sample charts</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        if "comfort_violation_minutes" in ok.columns:
            ch = alt.Chart(ok).mark_bar().encode(
                x=alt.X("sample_id:N", title="Sample"),
                y=alt.Y("comfort_violation_minutes:Q", title="Comfort violation (min)"),
                color=alt.Color("mode:N", title="Mode"),
                tooltip=["sample_id", "parser_mode_requested", "mode", "comfort_violation_minutes", "dr_compliance_score", "hvac_on_hours"]
            ).properties(height=320)
            st.altair_chart(ch, use_container_width=True)

    with c2:
        if "dr_compliance_score" in ok.columns:
            ch = alt.Chart(ok).mark_bar().encode(
                x=alt.X("sample_id:N", title="Sample"),
                y=alt.Y("dr_compliance_score:Q", title="DR compliance", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("mode:N", title="Mode"),
                tooltip=["sample_id", "parser_mode_requested", "mode", "dr_compliance_score", "comfort_violation_minutes"]
            ).properties(height=320)
            st.altair_chart(ch, use_container_width=True)

    st.markdown("<div class='section'>Tradeoff view</div>", unsafe_allow_html=True)
    if {"comfort_violation_minutes", "dr_compliance_score"}.issubset(ok.columns):
        sc = alt.Chart(ok).mark_circle(size=120).encode(
            x=alt.X("comfort_violation_minutes:Q", title="Comfort violation (min)"),
            y=alt.Y("dr_compliance_score:Q", title="DR compliance", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("mode:N", title="Mode"),
            shape=alt.Shape("parser_mode_requested:N", title="Parser"),
            tooltip=["sample_id", "parser_mode_requested", "mode", "comfort_violation_minutes", "dr_compliance_score", "hvac_on_hours", "parser_fallback"]
        ).properties(height=360).interactive()
        st.altair_chart(sc, use_container_width=True)

    if summary is not None and not summary.empty:
        st.markdown("<div class='section'>Summary comparison</div>", unsafe_allow_html=True)
        summary = summary.copy()
        summary["config"] = summary["parser_mode_requested"].astype(str) + " | " + summary["mode"].astype(str)

        mcols = [c for c in ["feasibility_rate", "avg_comfort_violation_minutes", "avg_dr_compliance_score", "avg_hvac_on_hours"] if c in summary.columns]
        if mcols:
            mdf = summary[["config"] + mcols].melt("config", var_name="metric", value_name="value")
            bc = alt.Chart(mdf).mark_bar().encode(
                x=alt.X("config:N", title="Configuration"),
                y=alt.Y("value:Q", title="Value"),
                color=alt.Color("metric:N", title="Metric"),
                column=alt.Column("metric:N", title=None),
                tooltip=["config", "metric", "value"]
            ).properties(height=260)
            st.altair_chart(bc, use_container_width=True)


# -----------------------------
# Session state for results + saved file paths
# -----------------------------
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()
if "summary_df" not in st.session_state:
    st.session_state.summary_df = pd.DataFrame()
if "saved_paths" not in st.session_state:
    st.session_state.saved_paths = None


# -----------------------------
# Pages
# -----------------------------
if page == "Dashboard":
    st.markdown("<div class='section'>Run benchmark</div>", unsafe_allow_html=True)

    if run_selected:
        try:
            df, summary = run_benchmark(parser_mode, enforce_guest_hard_comfort)
            st.session_state.results_df = df
            st.session_state.summary_df = summary

            # Fix 1: ALWAYS write outputs after run
            prefix = f"benchmark_{parser_mode}_{'hard' if enforce_guest_hard_comfort else 'soft'}"
            r_csv, s_csv, r_json = save_outputs(df, summary, prefix=prefix)
            st.session_state.saved_paths = (r_csv, s_csv, r_json)

            st.success("Run complete. Outputs saved.")
        except Exception as e:
            st.exception(e)

    if run_full:
        try:
            df, summary = run_comparison_all()
            st.session_state.results_df = df
            st.session_state.summary_df = summary

            # Fix 1: ALWAYS write outputs after run
            prefix = "comparison_all"
            r_csv, s_csv, r_json = save_outputs(df, summary, prefix=prefix)
            st.session_state.saved_paths = (r_csv, s_csv, r_json)

            st.success("Comparison complete. Outputs saved.")
        except Exception as e:
            st.exception(e)

    df = st.session_state.results_df
    summary = st.session_state.summary_df

    if df is None or df.empty:
        st.info("Run a benchmark to see KPI cards, graphs, tables, and downloadable outputs.")
    else:
        # Fix 2: Downloads are always available
        st.markdown("<div class='section'>Download outputs</div>", unsafe_allow_html=True)
        download_buttons(df, summary, prefix="nl_hems")

        # Show saved file paths (helps confirm Fix 1)
        if st.session_state.saved_paths:
            r_csv, s_csv, r_json = st.session_state.saved_paths
            st.caption(f"Saved files on server: {r_csv.name}, {r_json.name}" + (f", {s_csv.name}" if s_csv else ""))

        kpi_cards(df)
        charts(df, summary)

        st.markdown("<div class='section'>Tables</div>", unsafe_allow_html=True)
        if summary is not None and not summary.empty:
            st.write("Summary")
            st.dataframe(summary, use_container_width=True)
        st.write("Detailed results")
        st.dataframe(df, use_container_width=True)

elif page == "Playground":
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

            mapped = map_fuzzy_preferences(intent, enforce_guest_hard_comfort=play_hard)
            sched = optimize_schedule(mapped, context=override_context if enable_overrides else {})

            st.write({"parser_used": meta.get("parser_used"), "fallback": meta.get("fallback"), "error": meta.get("error")})
            st.json(intent.model_dump())
            st.write({"clarification_pred": clar, "clarification_question": q})
            st.json(mapped)
            st.json(sched)

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

        except Exception as e:
            st.exception(e)

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
            ok = 0
            bad = []
            for i, line in enumerate(edited.splitlines(), start=1):
                s = line.strip()
                if not s:
                    continue
                try:
                    json.loads(s)
                    ok += 1
                except Exception as e:
                    bad.append((i, str(e), s[:120]))
            if bad:
                st.error(f"Invalid lines: {len(bad)}")
                st.write(bad[:10])
            else:
                st.success(f"All good. Parsed {ok} JSON lines.")

        st.info("Tip: Add per-sample context like "
                "{\"context\":{\"initial_temp_c\":21,\"dr_event_hours\":[19,20],\"price\":{\"16\":2,...}}}")

elif page == "Files":
    st.markdown("<div class='section'>Outputs</div>", unsafe_allow_html=True)

    out_dir = get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(out_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not files:
        st.info("No output files yet. Run a benchmark from the Dashboard first.")
    else:
        df_files = pd.DataFrame([{
            "name": f.name,
            "size": f.stat().st_size,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds"),
            "path": str(f)
        } for f in files])

        st.dataframe(df_files, use_container_width=True)

        pick = st.selectbox("Preview", df_files["name"].tolist())
        fp = out_dir / pick

        if fp.suffix.lower() == ".csv":
            st.dataframe(pd.read_csv(fp).head(300), use_container_width=True)
        elif fp.suffix.lower() == ".json":
            content = json.loads(fp.read_text(encoding="utf-8"))
            st.json(content[:10] if isinstance(content, list) else content)
        else:
            st.write(fp.read_text(encoding="utf-8", errors="replace")[:3000])
