import os
import sys
import json
import inspect
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
from llm_parser import parse_command_by_mode
from clarification import should_clarify
from core import map_fuzzy_preferences, optimize_schedule


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
</style>
""", unsafe_allow_html=True)

st.title("⚡ NL-HEMS-DR Benchmark Dashboard")
st.caption("Natural language → intent/fuzzy mapping → OR-Tools optimization → metrics → exports")


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

    # section-based fallback
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
    ROOT / "data" / "outputs",             # may be read-only on Streamlit Cloud
    Path("/tmp") / "nl-hems-outputs",      # usually writable on Streamlit Cloud
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
# Sidebar: run settings + context overrides
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
    heat_c = st.slider("Heating gain (°C/hour ON)", 0.0, 2.0, 0.8, 0.05)

    # DR hours
    hours_list = list(range(h_start, h_end))
    dr_hours = st.multiselect("DR event hours", options=hours_list, default=[h for h in [18, 19] if h in hours_list])

    # Price table (editable)
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


# -----------------------------
# Compatibility wrapper: run_one_config may or may not accept override_context/use_sample_context
# -----------------------------
def run_one_config_compat(samples, parser_mode: str, comfort_mode: dict, override_ctx: dict | None, use_sample_ctx: bool):
    sig = inspect.signature(run_one_config)
    kwargs = {}

    if "override_context" in sig.parameters:
        kwargs["override_context"] = override_ctx
    if "use_sample_context" in sig.parameters:
        kwargs["use_sample_context"] = use_sample_ctx

    return run_one_config(samples, parser_mode=parser_mode, comfort_mode=comfort_mode, **kwargs)


# -----------------------------
# Run actions (works from any page)
# -----------------------------
def run_selected_benchmark():
    samples = load_jsonl(Path(dataset_path))
    cmode = {"name": "hard_guest_comfort" if enforce_hard else "soft_comfort",
             "enforce_guest_hard_comfort": enforce_hard}

    rows = run_one_config_compat(
        samples,
        parser_mode=parser_mode,
        comfort_mode=cmode,
        override_ctx=override_context,
        use_sample_ctx=use_sample_context,
    )

    df = pd.DataFrame(rows)
    summary = build_summary(df)

    st.session_state.results_df = df
    st.session_state.summary_df = summary

    out_dir, r_csv, s_csv, r_json = save_outputs(df, summary, prefix=f"benchmark_{parser_mode}_{cmode['name']}")
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

    for pmode, hard in configs:
        cmode = {"name": "hard_guest_comfort" if hard else "soft_comfort",
                 "enforce_guest_hard_comfort": hard}
        all_rows.extend(
            run_one_config_compat(
                samples,
                parser_mode=pmode,
                comfort_mode=cmode,
                override_ctx=override_context,
                use_sample_ctx=use_sample_context,
            )
        )

    df = pd.DataFrame(all_rows)
    summary = build_summary(df)

    st.session_state.results_df = df
    st.session_state.summary_df = summary

    out_dir, r_csv, s_csv, r_json = save_outputs(df, summary, prefix="comparison_all")
    st.session_state.saved_info = {
        "out_dir": str(out_dir),
        "results_csv": str(r_csv),
        "summary_csv": str(s_csv) if s_csv else None,
        "results_json": str(r_json),
    }


# Execute run immediately, then jump to Dashboard
if run_selected:
    try:
        run_selected_benchmark()
        st.success("Benchmark run complete. Outputs saved.")
        st.session_state.page_stack = ["Dashboard"]
        st.rerun()
    except Exception as e:
        st.exception(e)

if run_full:
    try:
        run_full_comparison()
        st.success("Comparison run complete. Outputs saved.")
        st.session_state.page_stack = ["Dashboard"]
        st.rerun()
    except Exception as e:
        st.exception(e)


# -----------------------------
# Charts (Tradeoff fix included)
# -----------------------------
def kpi_cards(df: pd.DataFrame):
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
            if pd.notna(val)
            else "<div class='card'><div class='small'>Avg comfort violation (min)</div><div class='big'>n/a</div></div>",
            unsafe_allow_html=True
        )
    else:
        c3.markdown("<div class='card'><div class='small'>Avg comfort violation (min)</div><div class='big'>n/a</div></div>", unsafe_allow_html=True)

    if len(ok) and "dr_compliance_score" in ok.columns:
        ok["dr_compliance_score"] = pd.to_numeric(ok["dr_compliance_score"], errors="coerce")
        val = ok["dr_compliance_score"].mean()
        c4.markdown(
            f"<div class='card'><div class='small'>Avg DR compliance</div><div class='big'>{val:.2f}</div></div>"
            if pd.notna(val)
            else "<div class='card'><div class='small'>Avg DR compliance</div><div class='big'>n/a</div></div>",
            unsafe_allow_html=True
        )
    else:
        c4.markdown("<div class='card'><div class='small'>Avg DR compliance</div><div class='big'>n/a</div></div>", unsafe_allow_html=True)


def charts(df: pd.DataFrame):
    plot_df = df.copy()

    # Normalize and filter OK
    if "status" in plot_df.columns:
        plot_df["status"] = plot_df["status"].astype(str).str.strip().str.lower()
        plot_df = plot_df[plot_df["status"] == "ok"].copy()

    if plot_df.empty:
        st.warning("No rows with status='ok' to plot. Check errors in the table.")
        return

    # Ensure required columns
    if "mode" not in plot_df.columns:
        plot_df["mode"] = "run"
    if "parser_mode_requested" not in plot_df.columns:
        plot_df["parser_mode_requested"] = "unknown"
    if "sample_id" not in plot_df.columns:
        plot_df["sample_id"] = plot_df.index.astype(str)

    # Coerce numerics (MAIN FIX)
    for col in ["comfort_violation_minutes", "dr_compliance_score", "hvac_on_hours"]:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

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

    st.markdown("<div class='section'>Tradeoff</div>", unsafe_allow_html=True)

    needed = {"comfort_violation_minutes", "dr_compliance_score"}
    if not needed.issubset(plot_df.columns):
        st.error(f"Missing columns for tradeoff plot: {needed - set(plot_df.columns)}")
        st.info("Make sure your runner produces comfort_violation_minutes and dr_compliance_score.")
        return

    scatter_df = plot_df.dropna(subset=["comfort_violation_minutes", "dr_compliance_score"]).copy()

    if scatter_df.empty:
        st.warning(
            "Tradeoff plot has no points because comfort_violation_minutes or dr_compliance_score are empty/non-numeric.\n"
            "Open the results table and verify those columns contain numbers."
        )
        with st.expander("Debug (types + sample rows)"):
            st.write(plot_df[["sample_id", "comfort_violation_minutes", "dr_compliance_score", "status"]].head(20))
            st.write(plot_df.dtypes)
        return

    sc = alt.Chart(scatter_df).mark_circle(size=120).encode(
        x=alt.X("comfort_violation_minutes:Q", title="Comfort violation (min)"),
        y=alt.Y("dr_compliance_score:Q", title="DR compliance", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("mode:N", title="Mode"),
        shape=alt.Shape("parser_mode_requested:N", title="Parser"),
        tooltip=["sample_id", "parser_mode_requested", "mode",
                 "comfort_violation_minutes", "dr_compliance_score", "hvac_on_hours", "parser_fallback"],
    ).properties(height=360).interactive()

    st.altair_chart(sc, use_container_width=True)


# -----------------------------
# Pages
# -----------------------------
if page == "Dashboard":
    df = st.session_state.results_df
    summary = st.session_state.summary_df

    if df is None or df.empty:
        st.info("Click **Run Selected Benchmark** in the sidebar to generate results and outputs.")
    else:
        st.markdown("<div class='section'>Downloads</div>", unsafe_allow_html=True)
        download_buttons(df, summary, prefix="nl_hems")

        if st.session_state.saved_info:
            st.caption(f"Saved on server in: {st.session_state.saved_info['out_dir']}")
            st.caption(f"CSV: {st.session_state.saved_info['results_csv']}")

        kpi_cards(df)
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
        st.dataframe(files_df, use_container_width=True)
        pick = st.selectbox("Preview", files_df["path"].tolist())
        fp = Path(pick)

        if fp.suffix.lower() == ".csv":
            st.dataframe(pd.read_csv(fp).head(300), use_container_width=True)
        elif fp.suffix.lower() == ".json":
            content = json.loads(fp.read_text(encoding="utf-8"))
            st.json(content[:10] if isinstance(content, list) else content)
        else:
            st.write(fp.read_text(encoding="utf-8", errors="replace")[:3000])

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
                    bad.append((i, str(e), s[:160]))
            if bad:
                st.error(f"Invalid lines: {len(bad)}")
                st.write(bad[:10])
            else:
                st.success(f"All good. Parsed {ok} JSON lines.")

        st.info(
            "Tip: Add per-sample context like "
            "{\"context\":{\"initial_temp_c\":21,\"dr_event_hours\":[19,20],\"price\":{\"16\":2,...}}}"
        )

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
            ctx = override_context if enable_overrides else {}
            sched = optimize_schedule(mapped, context=ctx)

            st.write({"parser_used": meta.get("parser_used"), "fallback": meta.get("fallback"), "error": meta.get("error")})
            st.write({"clarification_pred": clar, "clarification_question": q})
            st.json(intent.model_dump())
            st.json(mapped)

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
