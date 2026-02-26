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


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s in {"EOF", "PY", "```", "```json"}:
                continue
            if s.startswith("python3 - <<"):
                continue
            rows.append(json.loads(s))
    return rows


def merge_context(sample_ctx: dict | None, override_ctx: dict | None) -> dict:
    out = dict(sample_ctx or {})
    for k, v in (override_ctx or {}).items():
        if v is None:
            continue
        out[k] = v
    return out


def run_one_config(
    samples,
    parser_mode: str,
    comfort_mode: dict,
    override_context: dict | None = None,
    use_sample_context: bool = True,
):
    results = []
    enforce_guest_hard_comfort = comfort_mode["enforce_guest_hard_comfort"]
    comfort_mode_name = comfort_mode["name"]

    for sample in samples:
        sample_id = sample.get("sample_id", "UNKNOWN")
        command = sample.get("command", "")
        gold = sample.get("gold", {})

        sample_ctx = sample.get("context", {}) if use_sample_context else {}
        ctx = merge_context(sample_ctx, override_context)

        try:
            intent, parser_meta = parse_command_by_mode(command, parser_mode)

            clarification_pred, clarification_question = should_clarify(command, intent)
            intent.clarification_needed = clarification_pred

            mapped = map_fuzzy_preferences(intent, enforce_guest_hard_comfort=enforce_guest_hard_comfort)
            sched = optimize_schedule(mapped, context=ctx)

            row = {
                "sample_id": sample_id,
                "status": sched.get("status", "error"),
                "command": command,
                "parser_mode_requested": parser_mode,
                "parser_used": parser_meta.get("parser_used"),
                "parser_fallback": parser_meta.get("fallback"),
                "parser_error": parser_meta.get("error"),
                "mode": comfort_mode_name,
                "enforce_guest_hard_comfort": enforce_guest_hard_comfort,

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
                row["comfort_violation_minutes"] = comfort_violation_minutes(schedule, mapped["min_temp_c"])
                row["dr_compliance_score"] = dr_compliance_score(schedule)
                row["hvac_on_hours"] = sum(r["hvac_on"] for r in schedule)
                row["avg_temp_c"] = round(sum(r["temp_c"] for r in schedule) / len(schedule), 2)
                row["context_used"] = str(sched.get("context_used", {}))
            else:
                row["comfort_violation_minutes"] = None
                row["dr_compliance_score"] = None
                row["hvac_on_hours"] = None
                row["avg_temp_c"] = None
                row["context_used"] = None

        except Exception as e:
            row = {"sample_id": sample_id, "status": "error", "command": command, "error": str(e)}

        results.append(row)

    return results


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    grouped = df.groupby(["parser_mode_requested", "mode"], dropna=False)

    for (parser_mode_requested, mode), g in grouped:
        total = len(g)
        ok = g[g["status"] == "ok"].copy()
        rec = {
            "parser_mode_requested": parser_mode_requested,
            "mode": mode,
            "samples": total,
            "feasibility_rate": len(ok) / total if total else 0.0,
        }

        if len(ok):
            rec["avg_comfort_violation_minutes"] = ok["comfort_violation_minutes"].mean()
            rec["avg_dr_compliance_score"] = ok["dr_compliance_score"].mean()
            rec["avg_hvac_on_hours"] = ok["hvac_on_hours"].mean()

            gm = ok["guest_event_match"].dropna()
            rec["guest_event_detection_accuracy"] = gm.mean() if len(gm) else None

            eval_df = ok.dropna(subset=["clarification_gold"]).copy()
            if len(eval_df):
                tp = ((eval_df["clarification_pred"] == True) & (eval_df["clarification_gold"] == True)).sum()
                fp = ((eval_df["clarification_pred"] == True) & (eval_df["clarification_gold"] == False)).sum()
                fn = ((eval_df["clarification_pred"] == False) & (eval_df["clarification_gold"] == True)).sum()
                tn = ((eval_df["clarification_pred"] == False) & (eval_df["clarification_gold"] == False)).sum()

                rec["clarification_accuracy"] = (tp + tn) / len(eval_df)
                rec["clarification_precision"] = tp / (tp + fp) if (tp + fp) else 0.0
                rec["clarification_recall"] = tp / (tp + fn) if (tp + fn) else 0.0

            if "parser_fallback" in ok.columns:
                pf = ok["parser_fallback"].fillna(False)
                rec["parser_fallback_rate"] = (pf == True).mean()

        rows.append(rec)

    return pd.DataFrame(rows)