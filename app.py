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


# -----------------------------
# Page setup + styling
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
st.caption("Natural-language → fuzzy preferences → optimized schedule → metrics → exports")


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
# Output saving (Fix 1) + downloads (Fix 2)
# -----------------------------
OUTPUT_DIR_CANDIDATES = [
    ROOT / "data" / "outputs",             # may be read-only on Community Cloud
    Path("/tmp") / "nl-hems-outputs",      # usually writable
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
    raise RuntimeError("No writable output directory found (tried repo data/outputs and /tmp).")

def save_outputs(results_df: pd.DataFrame, summary_df: pd.DataFrame, prefix: str):
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

def download_buttons(results_df: pd.DataFrame, summary_df: pd.DataFrame, prefix: str):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "⬇️ Results CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{prefix}_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "⬇️ Results JSON",
            data=json.dumps(results_df.to_dict("records"), indent=2).encode("utf-8"),
            file_name=f"{prefix}_results.json",
            mime="application/json",
            use_container_width=True,
        )
    with c3:
        if summary_df is not None and not summary_df.empty:
            st.download_button(
                "⬇️ Summary CSV",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{prefix}_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.download_button(
                "⬇️ Summary CSV",
                data=b"",
                file_name=f"{prefix}_summary.csv",
                mime="text/csv",
                disabled=True,
                use_container_width=True,
            )

def list_output_files():
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
# Sidebar controls + Run buttons
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
# Run actions (IMPORTANT: now runs from ANY page)
# -----------------------------
def run_selected_benchmark():
    samples = load_jsonl(Path(dataset_path))
    cmode = {"name": "hard_guest_comfort" if enforce_hard else "soft_comfort",
             "enforce_guest_hard_comfort": enforce_hard}
    rows = run_one_config(samples, parser_mode=parser_mode, comfort_mode=cmode)
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
        all_rows.extend(run_one_config(samples, parser_mode=pmode, comfort_mode=cmode))

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

# Execute runs even if user is on Files/Playground page
if run_selected:
    try:
        run_selected_benchmark()
        st.success("Benchmark run complete. Outputs saved.")
        # Auto-jump user back to Dashboard so they see results
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
# Dashboard page
# -----------------------------
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

def charts(df: pd.DataFrame):
    ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()
    if ok.empty:
        st.warning("No successful rows to chart.")
        return

    c1, c2 = st.columns(2)
    with c1:
        if "comfort_violation_minutes" in ok.columns:
            ch = alt.Chart(ok).mark_bar().encode(
                x=alt.X("sample_id:N", title="Sample"),
                y=alt.Y("comfort_violation_minutes:Q", title="Comfort violation (min)"),
                color=alt.Color("mode:N", title="Mode"),
                tooltip=["sample_id", "parser_mode_requested", "mode", "comfort_violation_minutes", "dr_compliance_score"]
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

    if {"comfort_violation_minutes", "dr_compliance_score"}.issubset(ok.columns):
        st.markdown("<div class='section'>Tradeoff</div>", unsafe_allow_html=True)
        sc = alt.Chart(ok).mark_circle(size=120).encode(
            x=alt.X("comfort_violation_minutes:Q", title="Comfort violation (min)"),
            y=alt.Y("dr_compliance_score:Q", title="DR compliance", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("mode:N", title="Mode"),
            shape=alt.Shape("parser_mode_requested:N", title="Parser"),
            tooltip=["sample_id", "parser_mode_requested", "mode", "comfort_violation_minutes", "dr_compliance_score", "parser_fallback"]
        ).properties(height=360).interactive()
        st.altair_chart(sc, use_container_width=True)

if page == "Dashboard":
    df = st.session_state.results_df
    summary = st.session_state.summary_df

    if df is None or df.empty:
        st.info("Click **Run Selected Benchmark** (left sidebar) to generate outputs.")
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
        st.warning("No output files yet. Run a benchmark from the sidebar (it works from any page now).")
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
                    bad.append((i, str(e), s[:120]))
            if bad:
                st.error(f"Invalid lines: {len(bad)}")
                st.write(bad[:10])
            else:
                st.success(f"All good. Parsed {ok} JSON lines.")

elif page == "Playground":
    st.info("Playground kept minimal in this version. Use Dashboard + Files for outputs.")
