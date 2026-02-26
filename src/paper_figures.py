import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def latest_file(folder: Path, pattern: str):
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_data(results_csv: Path | None, summary_csv: Path | None):
    out_dir = Path("data/outputs")

    if results_csv is None:
        results_csv = latest_file(out_dir, "comparison_results*.csv") or latest_file(out_dir, "*results*.csv")
    if summary_csv is None:
        summary_csv = latest_file(out_dir, "comparison_summary*.csv") or latest_file(out_dir, "*summary*.csv")

    if results_csv is None or not Path(results_csv).exists():
        raise FileNotFoundError("Could not find results CSV. Pass --results explicitly.")
    if summary_csv is None or not Path(summary_csv).exists():
        raise FileNotFoundError("Could not find summary CSV. Pass --summary explicitly.")

    results_df = pd.read_csv(results_csv)
    summary_df = pd.read_csv(summary_csv)
    return results_df, summary_df, Path(results_csv), Path(summary_csv)


def add_config_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parser_col = "parser_mode_requested" if "parser_mode_requested" in out.columns else None
    mode_col = "mode" if "mode" in out.columns else None

    if parser_col and mode_col:
        out["config_label"] = out[parser_col].astype(str) + " | " + out[mode_col].astype(str)
    elif mode_col:
        out["config_label"] = out[mode_col].astype(str)
    else:
        out["config_label"] = "config"
    return out


def save_fig(fig, out_path: Path, dpi=300):
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_tradeoff_scatter(results_df: pd.DataFrame, out_dir: Path):
    if results_df.empty:
        return

    df = results_df.copy()
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    if df.empty:
        return

    required = {"comfort_violation_minutes", "dr_compliance_score"}
    if not required.issubset(df.columns):
        return

    df = add_config_label(df)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    modes = sorted(df["config_label"].dropna().unique().tolist())

    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    for i, label in enumerate(modes):
        sub = df[df["config_label"] == label]
        ax.scatter(
            sub["comfort_violation_minutes"],
            sub["dr_compliance_score"],
            label=label,
            marker=markers[i % len(markers)],
            s=55,
            alpha=0.85,
        )

    for _, r in df.iterrows():
        sid = str(r.get("sample_id", ""))
        if sid:
            ax.annotate(
                sid,
                (r["comfort_violation_minutes"], r["dr_compliance_score"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )

    ax.set_title("Comfort–DR Tradeoff by Sample")
    ax.set_xlabel("Comfort Violation (minutes)")
    ax.set_ylabel("DR Compliance Score")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    save_fig(fig, out_dir / "fig_tradeoff_scatter.png")


def plot_per_sample_bars(results_df: pd.DataFrame, out_dir: Path):
    if results_df.empty:
        return

    df = results_df.copy()
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    if df.empty:
        return

    if not {"sample_id", "comfort_violation_minutes", "dr_compliance_score"}.issubset(df.columns):
        return

    df = add_config_label(df)
    # Sort for stable figure ordering
    df["sample_id"] = df["sample_id"].astype(str)
    df = df.sort_values(["sample_id", "config_label"]).reset_index(drop=True)
    df["sample_cfg"] = df["sample_id"] + " | " + df["config_label"]

    # Comfort violation bar
    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(df)), 4.8))
    ax.bar(df["sample_cfg"], df["comfort_violation_minutes"])
    ax.set_title("Per-Sample Comfort Violation")
    ax.set_ylabel("Comfort Violation (minutes)")
    ax.set_xlabel("Sample | Config")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=70, labelsize=7)
    save_fig(fig, out_dir / "fig_per_sample_comfort_violation.png")

    # DR compliance bar
    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(df)), 4.8))
    ax.bar(df["sample_cfg"], df["dr_compliance_score"])
    ax.set_title("Per-Sample DR Compliance")
    ax.set_ylabel("DR Compliance Score")
    ax.set_xlabel("Sample | Config")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=70, labelsize=7)
    save_fig(fig, out_dir / "fig_per_sample_dr_compliance.png")


def plot_summary_metric_bars(summary_df: pd.DataFrame, out_dir: Path):
    if summary_df.empty:
        return

    df = add_config_label(summary_df)

    metric_specs = [
        ("feasibility_rate", "Feasibility Rate", (0, 1.02), "fig_summary_feasibility.png"),
        ("avg_comfort_violation_minutes", "Avg Comfort Violation (minutes)", None, "fig_summary_avg_comfort_violation.png"),
        ("avg_dr_compliance_score", "Avg DR Compliance Score", (0, 1.02), "fig_summary_avg_dr_compliance.png"),
        ("avg_hvac_on_hours", "Avg HVAC On-Hours", None, "fig_summary_avg_hvac_on_hours.png"),
        ("parser_fallback_rate", "Parser Fallback Rate", (0, 1.02), "fig_summary_parser_fallback_rate.png"),
    ]

    for col, title, ylim, fname in metric_specs:
        if col not in df.columns:
            continue
        vals = df[["config_label", col]].dropna()
        if vals.empty:
            continue

        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        ax.bar(vals["config_label"], vals[col])
        ax.set_title(title)
        ax.set_xlabel("Configuration")
        ax.set_ylabel(title)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20, labelsize=9)
        save_fig(fig, out_dir / fname)


def plot_clarification_metrics(summary_df: pd.DataFrame, out_dir: Path):
    needed = ["clarification_accuracy", "clarification_precision", "clarification_recall"]
    if not all(c in summary_df.columns for c in needed):
        return

    df = add_config_label(summary_df).copy()
    df = df[["config_label"] + needed].dropna(how="all", subset=needed)
    if df.empty:
        return

    x = range(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar([i - width for i in x], df["clarification_accuracy"].fillna(0), width=width, label="Accuracy")
    ax.bar(x, df["clarification_precision"].fillna(0), width=width, label="Precision")
    ax.bar([i + width for i in x], df["clarification_recall"].fillna(0), width=width, label="Recall")

    ax.set_title("Clarification Policy Metrics")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["config_label"], rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    save_fig(fig, out_dir / "fig_clarification_metrics.png")


def plot_guest_metrics(summary_df: pd.DataFrame, out_dir: Path):
    df = add_config_label(summary_df).copy()

    # Guest comfort violation
    if "guest_avg_comfort_violation_minutes" in df.columns:
        vals = df[["config_label", "guest_avg_comfort_violation_minutes"]].dropna()
        if not vals.empty:
            fig, ax = plt.subplots(figsize=(8.4, 4.6))
            ax.bar(vals["config_label"], vals["guest_avg_comfort_violation_minutes"])
            ax.set_title("Guest-Scenario Avg Comfort Violation")
            ax.set_ylabel("Minutes")
            ax.set_xlabel("Configuration")
            ax.grid(True, axis="y", alpha=0.25)
            ax.tick_params(axis="x", rotation=20)
            save_fig(fig, out_dir / "fig_guest_avg_comfort_violation.png")

    # Guest DR compliance
    if "guest_avg_dr_compliance_score" in df.columns:
        vals = df[["config_label", "guest_avg_dr_compliance_score"]].dropna()
        if not vals.empty:
            fig, ax = plt.subplots(figsize=(8.4, 4.6))
            ax.bar(vals["config_label"], vals["guest_avg_dr_compliance_score"])
            ax.set_title("Guest-Scenario Avg DR Compliance")
            ax.set_ylabel("Score")
            ax.set_xlabel("Configuration")
            ax.set_ylim(0, 1.02)
            ax.grid(True, axis="y", alpha=0.25)
            ax.tick_params(axis="x", rotation=20)
            save_fig(fig, out_dir / "fig_guest_avg_dr_compliance.png")

    # Guest min temp
    if "guest_window_min_temp_c_avg" in df.columns:
        vals = df[["config_label", "guest_window_min_temp_c_avg"]].dropna()
        if not vals.empty:
            fig, ax = plt.subplots(figsize=(8.4, 4.6))
            ax.bar(vals["config_label"], vals["guest_window_min_temp_c_avg"])
            ax.set_title("Guest-Window Minimum Temperature (Avg)")
            ax.set_ylabel("Temperature (°C)")
            ax.set_xlabel("Configuration")
            ax.grid(True, axis="y", alpha=0.25)
            ax.tick_params(axis="x", rotation=20)
            save_fig(fig, out_dir / "fig_guest_window_min_temp_avg.png")


def write_manifest(out_dir: Path, results_csv: Path, summary_csv: Path):
    pngs = sorted([p.name for p in out_dir.glob("*.png")])
    manifest = {
        "results_csv": str(results_csv),
        "summary_csv": str(summary_csv),
        "generated_pngs": pngs,
    }
    (out_dir / "figures_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved: {out_dir / 'figures_manifest.json'}")


def main():
    parser = argparse.ArgumentParser(description="Generate PNG figures from NL-HEMS-DR benchmark outputs.")
    parser.add_argument("--results", type=str, default=None, help="Path to detailed results CSV")
    parser.add_argument("--summary", type=str, default=None, help="Path to summary CSV")
    parser.add_argument("--outdir", type=str, default="paper_figures", help="Output directory for PNG files")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    ensure_dir(out_dir)

    results_df, summary_df, results_csv, summary_csv = load_data(
        Path(args.results) if args.results else None,
        Path(args.summary) if args.summary else None,
    )

    print(f"Using results: {results_csv}")
    print(f"Using summary: {summary_csv}")
    print(f"Writing figures to: {out_dir.resolve()}")

    plot_tradeoff_scatter(results_df, out_dir)
    plot_per_sample_bars(results_df, out_dir)
    plot_summary_metric_bars(summary_df, out_dir)
    plot_clarification_metrics(summary_df, out_dir)
    plot_guest_metrics(summary_df, out_dir)
    write_manifest(out_dir, results_csv, summary_csv)

    print("Done.")


if __name__ == "__main__":
    main()