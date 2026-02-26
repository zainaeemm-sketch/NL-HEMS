import json
from pathlib import Path
import pandas as pd

from core import (
    map_fuzzy_preferences,
    optimize_schedule,
    comfort_violation_minutes,
    dr_compliance_score,
)
from clarification import should_clarify
from llm_parser import parse_command_by_mode


# =========================
# Experiment toggles
# =========================
PARSER_MODE = "llm"   # "stub" or "llm"
ENFORCE_GUEST_HARD_COMFORT = True  # True = V3 mode


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            if s in {"EOF", "PY", "```", "```json"}:
                continue
            if s.startswith("python3 - <<"):
                continue

            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSONL at line {lineno}: {s[:120]!r} ({e})"
                ) from e
    return rows


def main():
    root = Path(__file__).resolve().parents[1]
    input_path = root / "data" / "samples" / "benchmark_samples.jsonl"
    out_csv = root / "data" / "outputs" / "benchmark_results.csv"
    out_json = root / "data" / "outputs" / "benchmark_results.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {input_path}")

    samples = load_jsonl(input_path)
    results = []

    for sample in samples:
        sample_id = sample.get("sample_id", "UNKNOWN")
        command = sample.get("command", "")
        gold = sample.get("gold", {})

        try:
            # 1) Parse (stub or LLM)
            intent, parser_meta = parse_command_by_mode(command, PARSER_MODE)

            # 2) Clarification decision
            clarification_pred, clarification_question = should_clarify(command, intent)
            intent.clarification_needed = clarification_pred

            # 3) Fuzzy mapping + scheduling
            mapped = map_fuzzy_preferences(
                intent,
                enforce_guest_hard_comfort=ENFORCE_GUEST_HARD_COMFORT
            )
            sched = optimize_schedule(mapped)

            row = {
                "sample_id": sample_id,
                "status": sched.get("status", "error"),
                "command": command,
                "parser_mode_requested": PARSER_MODE,
                "parser_used": parser_meta.get("parser_used"),
                "parser_fallback": parser_meta.get("fallback"),
                "parser_error": parser_meta.get("error"),
                "mode": "hard_guest_comfort" if ENFORCE_GUEST_HARD_COMFORT else "soft_comfort",
                "enforce_guest_hard_comfort": ENFORCE_GUEST_HARD_COMFORT,

                "guest_event_pred": intent.guest_event,
                "guest_event_gold": gold.get("expect_guest_event"),
                "guest_event_match": (
                    int(intent.guest_event == gold.get("expect_guest_event"))
                    if gold.get("expect_guest_event") is not None else None
                ),

                "clarification_pred": clarification_pred,
                "clarification_gold": gold.get("clarification_needed"),
                "clarification_match": (
                    int(clarification_pred == gold.get("clarification_needed"))
                    if gold.get("clarification_needed") is not None else None
                ),
                "clarification_question": clarification_question,

                "mapped_min_temp_c": mapped["min_temp_c"],
                "gold_min_temp_c": gold.get("min_temp_c"),
                "objective_value": sched.get("objective_value"),
                "notes": " | ".join(intent.notes) if intent.notes else "",
            }

            if sched.get("status") == "ok":
                schedule = sched["schedule"]
                row["comfort_violation_minutes"] = comfort_violation_minutes(
                    schedule, mapped["min_temp_c"]
                )
                row["dr_compliance_score"] = dr_compliance_score(schedule)
                row["hvac_on_hours"] = sum(r["hvac_on"] for r in schedule)
                row["avg_temp_c"] = round(sum(r["temp_c"] for r in schedule) / len(schedule), 2)

                guest_rows = [r for r in schedule if r.get("guest_window")]
                if guest_rows:
                    row["guest_window_avg_temp_c"] = round(
                        sum(r["temp_c"] for r in guest_rows) / len(guest_rows), 2
                    )
                    row["guest_window_min_temp_c"] = min(r["temp_c"] for r in guest_rows)
                else:
                    row["guest_window_avg_temp_c"] = None
                    row["guest_window_min_temp_c"] = None

                dr_rows = [r for r in schedule if r.get("dr_event")]
                row["dr_hvac_on_hours"] = sum(r["hvac_on"] for r in dr_rows) if dr_rows else None
            else:
                row["comfort_violation_minutes"] = None
                row["dr_compliance_score"] = None
                row["hvac_on_hours"] = None
                row["avg_temp_c"] = None
                row["guest_window_avg_temp_c"] = None
                row["guest_window_min_temp_c"] = None
                row["dr_hvac_on_hours"] = None

        except Exception as e:
            row = {
                "sample_id": sample_id,
                "status": "error",
                "command": command,
                "parser_mode_requested": PARSER_MODE,
                "mode": "hard_guest_comfort" if ENFORCE_GUEST_HARD_COMFORT else "soft_comfort",
                "enforce_guest_hard_comfort": ENFORCE_GUEST_HARD_COMFORT,
                "error": str(e),
            }

        results.append(row)

    df = pd.DataFrame(results)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== BENCHMARK COMPLETE ===")
    print(f"Samples run: {len(df)}")
    print(f"Parser mode requested: {PARSER_MODE}")
    print(f"Comfort mode: {'hard_guest_comfort' if ENFORCE_GUEST_HARD_COMFORT else 'soft_comfort'}")
    print(f"CSV:  {out_csv}")
    print(f"JSON: {out_json}")

    ok_df = df[df["status"] == "ok"] if "status" in df.columns else pd.DataFrame()

    print("\n=== SUMMARY ===")
    print(f"Feasibility rate: {len(ok_df)}/{len(df)} = {len(ok_df)/max(1, len(df)):.2f}")

    if len(ok_df):
        if "comfort_violation_minutes" in ok_df.columns:
            print(f"Avg comfort violation (min): {ok_df['comfort_violation_minutes'].mean():.1f}")
        if "dr_compliance_score" in ok_df.columns:
            print(f"Avg DR compliance score: {ok_df['dr_compliance_score'].mean():.2f}")
        if "hvac_on_hours" in ok_df.columns:
            print(f"Avg HVAC on-hours: {ok_df['hvac_on_hours'].mean():.2f}")
        if "guest_event_match" in ok_df.columns:
            g = ok_df["guest_event_match"].dropna()
            if len(g):
                print(f"Guest-event detection accuracy: {g.mean():.2f}")

        # Clarification metrics
        if "clarification_pred" in ok_df.columns and "clarification_gold" in ok_df.columns:
            eval_df = ok_df.dropna(subset=["clarification_gold"]).copy()
            if len(eval_df) > 0:
                tp = ((eval_df["clarification_pred"] == True) & (eval_df["clarification_gold"] == True)).sum()
                fp = ((eval_df["clarification_pred"] == True) & (eval_df["clarification_gold"] == False)).sum()
                fn = ((eval_df["clarification_pred"] == False) & (eval_df["clarification_gold"] == True)).sum()
                tn = ((eval_df["clarification_pred"] == False) & (eval_df["clarification_gold"] == False)).sum()

                precision = tp / (tp + fp) if (tp + fp) else 0.0
                recall = tp / (tp + fn) if (tp + fn) else 0.0
                acc = (tp + tn) / len(eval_df) if len(eval_df) else 0.0

                print(f"Clarification accuracy: {acc:.2f}")
                print(f"Clarification precision: {precision:.2f}")
                print(f"Clarification recall: {recall:.2f}")

        # Parser fallback info (useful in LLM mode)
        if "parser_fallback" in ok_df.columns:
            pf = ok_df["parser_fallback"].fillna(False)
            print(f"Parser fallback rate: {(pf == True).mean():.2f}")

    preview_cols = [c for c in [
        "sample_id", "status", "parser_mode_requested", "parser_used", "parser_fallback",
        "mode", "guest_event_pred", "guest_event_match",
        "clarification_pred", "clarification_gold", "clarification_match",
        "comfort_violation_minutes", "dr_compliance_score", "hvac_on_hours",
        "guest_window_min_temp_c"
    ] if c in df.columns]

    print("\n=== PREVIEW ===")
    print(df[preview_cols].to_string(index=False) if preview_cols else "No preview columns available.")


if __name__ == "__main__":
    main()