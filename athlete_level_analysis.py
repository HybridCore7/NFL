import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

DEFAULT_DATA_DIR = Path(r"D:\NFL\data")
DEFAULT_OUTPUT_DIR = Path(r"D:\NFL\outputs")
PRIMARY_POSITIONS = ["DB", "WR", "TE"]
EXCLUDED_THREE_CONE_FEATURES = {
    "three_cone_min_speed_cut",
    "three_cone_speed_retention",
    "three_cone_post_cut_reaccel",
}


def load_combined_inputs(data_dir: Path, output_dir: Path):
    feature_df = pd.read_csv(output_dir / "combine_movement_features.csv", low_memory=False)
    combine_results = pd.read_csv(data_dir / "combine_results.csv", low_memory=False)
    return feature_df, combine_results


def build_athlete_level_dataset(feature_df: pd.DataFrame, combine_results: pd.DataFrame) -> pd.DataFrame:
    primary = feature_df[~feature_df["feature_name"].isin(EXCLUDED_THREE_CONE_FEATURES)].copy()
    if primary.empty:
        raise ValueError("No primary movement features remain after excluding three-cone cut metrics.")

    primary = primary[["draft_year", "nfl_id", "feature_name", "feature_value", "feature_provenance"]].copy()
    primary = primary.dropna(subset=["feature_value"]).copy()

    athlete_summary = (
        primary.groupby(["draft_year", "nfl_id", "feature_name"], dropna=False)["feature_value"]
        .agg(["median", "count"])
        .reset_index()
        .rename(columns={"median": "feature_value_median", "count": "feature_value_valid_attempts"})
    )

    feature_wide = athlete_summary.pivot(index=["draft_year", "nfl_id"], columns="feature_name", values="feature_value_median").reset_index()
    feature_count = athlete_summary.pivot(index=["draft_year", "nfl_id"], columns="feature_name", values="feature_value_valid_attempts").reset_index()
    feature_wide.columns = [c if c in {"draft_year", "nfl_id"} else f"{c}_median" for c in feature_wide.columns]
    feature_count.columns = [c if c in {"draft_year", "nfl_id"} else f"{c}_n_valid" for c in feature_count.columns]

    summary = feature_wide.merge(feature_count, on=["draft_year", "nfl_id"], how="left")
    meta_cols = [
        "draft_year",
        "nfl_id",
        "combine_position",
        "combine_height",
        "combine_weight",
        "ten_yd_split",
        "forty",
        "vertical",
        "broad_jump",
        "three_cone",
        "short_shuttle",
        "bench_reps",
    ]
    meta = combine_results[meta_cols].drop_duplicates()
    summary = summary.merge(meta, on=["draft_year", "nfl_id"], how="left")

    provenance = (
        primary.groupby(["draft_year", "nfl_id"], dropna=False)["feature_provenance"]
        .agg(lambda x: "; ".join(sorted({str(v) for v in x if pd.notna(v)})))
        .reset_index(name="feature_provenance")
    )
    summary = summary.merge(provenance, on=["draft_year", "nfl_id"], how="left")

    summary["position_group"] = summary["combine_position"].apply(
        lambda pos: pos if pos in PRIMARY_POSITIONS else "OTHER"
    )
    return summary.sort_values(["draft_year", "nfl_id"]).reset_index(drop=True)


def build_coverage_report(athlete_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in athlete_df.columns if c.endswith("_median")]
    rows = []
    total_athletes = len(athlete_df)
    for col in feature_cols:
        valid = athlete_df[col].notna().sum()
        rows.append(
            {
                "feature_name": col.replace("_median", ""),
                "n_athletes_with_value": int(valid),
                "n_athletes_total": int(total_athletes),
                "coverage": float(valid / total_athletes) if total_athletes else 0.0,
                "missing_fraction": float(1.0 - valid / total_athletes) if total_athletes else 0.0,
            }
        )
    report = pd.DataFrame(rows).sort_values("feature_name").reset_index(drop=True)
    return report


def pearson_ci(x: np.ndarray, y: np.ndarray, n_bootstrap: int = 2000) -> tuple[float, float, float, float]:
    valid = ~(pd.isna(x) | pd.isna(y))
    x = np.asarray(x[valid], dtype=float)
    y = np.asarray(y[valid], dtype=float)
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan
    r, _ = pearsonr(x, y)
    z = 0.5 * math.log((1.0 + r) / (1.0 - r)) if abs(r) < 1.0 else np.sign(r) * 100.0
    se = 1.0 / math.sqrt(len(x) - 3)
    low = math.tanh(z - 1.96 * se)
    high = math.tanh(z + 1.96 * se)
    return float(r), float(low), float(high), float(len(x))


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, n_bootstrap: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    valid = ~(pd.isna(x) | pd.isna(y))
    x = np.asarray(x[valid], dtype=float)
    y = np.asarray(y[valid], dtype=float)
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    obs = []
    n = len(x)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        rho, _ = spearmanr(x[idx], y[idx])
        obs.append(rho)
    obs = np.asarray(obs, dtype=float)
    rho, _ = spearmanr(x, y)
    return float(rho), float(np.nanpercentile(obs, 2.5)), float(np.nanpercentile(obs, 97.5))


def build_position_summary(athlete_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    position_df = athlete_df[athlete_df["combine_position"].isin(PRIMARY_POSITIONS)].copy()
    for pos in PRIMARY_POSITIONS:
        pos_df = position_df[position_df["combine_position"] == pos].copy()
        total_unique = pos_df["nfl_id"].nunique()
        early = pos_df["early_accel_1s_median"].copy()
        forty = pd.to_numeric(pos_df["forty"], errors="coerce")
        corr_pair = pos_df[["forty", "early_accel_1s_median"]].dropna()
        n_valid = len(corr_pair)
        if n_valid >= 3:
            pearson_r, pearson_low, pearson_high, _ = pearson_ci(corr_pair["forty"].to_numpy(), corr_pair["early_accel_1s_median"].to_numpy())
            spearman_rho, spearman_low, spearman_high = bootstrap_spearman(
                corr_pair["forty"].to_numpy(), corr_pair["early_accel_1s_median"].to_numpy(), n_bootstrap=1000, seed=42
            )
        else:
            pearson_r = pearson_low = pearson_high = spearman_rho = spearman_low = spearman_high = np.nan

        early_missing = float(early.isna().mean()) if len(early) else np.nan
        rows.append(
            {
                "position_group": pos,
                "n_unique_athletes": int(total_unique),
                "n_valid_40_and_early_accel": int(n_valid),
                "pearson_r": pearson_r,
                "pearson_ci_low": pearson_low,
                "pearson_ci_high": pearson_high,
                "spearman_rho": spearman_rho,
                "spearman_ci_low": spearman_low,
                "spearman_ci_high": spearman_high,
                "early_accel_median": float(early.median()) if early.notna().any() else np.nan,
                "early_accel_q25": float(early.quantile(0.25)) if early.notna().any() else np.nan,
                "early_accel_q75": float(early.quantile(0.75)) if early.notna().any() else np.nan,
                "early_accel_iqr": float(early.quantile(0.75) - early.quantile(0.25)) if early.notna().any() else np.nan,
                "early_accel_missing_fraction": early_missing,
            }
        )
    return pd.DataFrame(rows)


def plot_athlete_level_scatter(athlete_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = athlete_df[athlete_df["combine_position"].isin(PRIMARY_POSITIONS)].copy()
    plot_df = plot_df.dropna(subset=["forty", "early_accel_1s_median"]).copy()
    colors = {"DB": "#1f77b4", "WR": "#ff7f0e", "TE": "#2ca02c"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, pos in zip(axes, PRIMARY_POSITIONS):
        subset = plot_df[plot_df["combine_position"] == pos].copy()
        x = pd.to_numeric(subset["forty"], errors="coerce")
        y = pd.to_numeric(subset["early_accel_1s_median"], errors="coerce")
        subset = subset.loc[~(x.isna() | y.isna())]
        ax.scatter(subset["forty"], subset["early_accel_1s_median"], s=30, alpha=0.8, color=colors[pos])
        ax.set_title(f"{pos} (n={len(subset)})")
        ax.set_xlabel("40-yard time (s)")
        ax.set_ylabel("Median early acceleration (yd/s^2)" if ax == axes[0] else "")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_outcome_audit() -> pd.DataFrame:
    career = pd.read_csv(DEFAULT_DATA_DIR / "player_career_successes.csv", low_memory=False)
    outcome_cols = [
        "career_offensive_snaps",
        "career_defensive_snaps",
        "career_special_teams_snaps",
        "career_games_active",
        "career_games_started",
        "ap_all_pro_1st_team",
        "ap_all_pro_2nd_team",
        "pro_bowl_original_ballot",
    ]
    rows = []
    for year in sorted(career["nfl_id"].unique()[:0]):
        pass
    for c in outcome_cols:
        values = pd.to_numeric(career[c], errors="coerce")
        rows.append(
            {
                "outcome": c,
                "n_non_missing": int(values.notna().sum()),
                "n_zero": int((values == 0).sum()),
                "n_positive": int((values > 0).sum()),
                "median": float(values.median()) if values.notna().any() else np.nan,
                "max": float(values.max()) if values.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_report(output_dir: Path, athlete_df: pd.DataFrame, corr_summary: pd.DataFrame, coverage_report: pd.DataFrame) -> None:
    report_path = output_dir / "athlete_level_analysis_report.txt"
    outcome_audit = build_outcome_audit()
    draft_year_counts = athlete_df["draft_year"].value_counts().sort_index()
    lines = [
        "Athlete-level movement analysis report",
        "====================================",
        "",
        "Dataset summary:",
        f"- Athletes retained: {len(athlete_df)}",
        f"- Draft-year counts: {dict(draft_year_counts.to_dict())}",
        "- Movement features were aggregated to athlete-level medians across valid attempts; repeated attempts were never treated as independent athletes.",
        "- Three-cone cut features were excluded from the primary athlete-level analysis because their real-data coverage is still limited.",
        "",
        "Position-level correlation summary:",
    ]
    for _, row in corr_summary.iterrows():
        lines.append(
            f"- {row['position_group']}: athletes={int(row['n_unique_athletes'])}, "
            f"pearson_r={row['pearson_r']:.3f} [{row['pearson_ci_low']:.3f}, {row['pearson_ci_high']:.3f}], "
            f"spearman_rho={row['spearman_rho']:.3f} [{row['spearman_ci_low']:.3f}, {row['spearman_ci_high']:.3f}], "
            f"median_early_accel={row['early_accel_median']:.3f}"
        )

    lines.extend([
        "",
        "Outcome-window audit:",
        "- The available labels in player_career_successes.csv are lifetime aggregate totals, not season-specific outcomes.",
        "- Draft classes have unequal exposure: 2023 has up to three observed seasons, 2024 up to two, and 2025 up to one.",
        "- Therefore, a common retrospective outcome window does not exist across the full 2023-2025 cohort without dropping the most recent classes.",
        "- The next valid predictive experiment should use a single, common observation window (for example: first two seasons for the 2023-2024 cohort, or first season for all 2023-2025 with explicit censoring), and all preprocessing should remain inside training folds.",
        "- Draft capital should remain a separate baseline; it represents market prior evaluation and should not be mixed with athletic measurements.",
        "",
        "Baseline definitions:",
        "1. Conventional Combine baseline: combine height, weight, 40-yard time, vertical, broad jump, ten-yard split, short shuttle, bench reps, etc.",
        "2. Draft-capital baseline: draft round and overall pick are held in a separate frame and never treated as athletic measurements.",
        "3. Conventional + movement baseline: conventional Combine features plus athlete-level movement summaries, excluding the three-cone cut metrics until their coverage is adequate.",
        "",
        "Outcome label summary:",
    ])
    for _, row in outcome_audit.iterrows():
        lines.append(
            f"- {row['outcome']}: non_missing={int(row['n_non_missing'])}, zero={int(row['n_zero'])}, positive={int(row['n_positive'])}, median={row['median']:.3f}, max={row['max']:.3f}"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_self_check(athlete_df: pd.DataFrame, output_dir: Path) -> None:
    required_files = [
        output_dir / "athlete_level_analysis.csv",
        output_dir / "athlete_level_coverage_report.csv",
        output_dir / "athlete_level_position_corr.csv",
        output_dir / "athlete_level_early_accel_scatter.png",
        output_dir / "athlete_level_analysis_report.txt",
    ]
    for p in required_files:
        if not p.exists():
            raise FileNotFoundError(f"Missing required output: {p.name}")
        if p.stat().st_size == 0:
            raise ValueError(f"Output is empty: {p.name}")

    if athlete_df.empty:
        raise ValueError("Athlete-level dataset is empty.")
    if athlete_df.duplicated(subset=["draft_year", "nfl_id"]).any():
        raise ValueError("Athlete-level dataset contains duplicate athlete rows.")
    if not {"nfl_id", "draft_year", "combine_position"}.issubset(athlete_df.columns):
        raise KeyError("Athlete-level dataset missing required identifiers.")
    if not athlete_df["nfl_id"].notna().all():
        raise ValueError("One or more athlete IDs are missing.")
    if athlete_df["forty"].notna().sum() == 0:
        raise ValueError("Forty-yard times are missing for all athletes.")
    print("SELF_CHECK_OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Athlete-level Combine movement analysis and baseline setup.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_df, combine_results = load_combined_inputs(args.data_dir, args.output_dir)
    athlete_df = build_athlete_level_dataset(feature_df, combine_results)
    coverage_report = build_coverage_report(athlete_df)

    corr_summary = build_position_summary(athlete_df)
    corr_summary.to_csv(args.output_dir / "athlete_level_position_corr.csv", index=False)

    plot_athlete_level_scatter(athlete_df, args.output_dir / "athlete_level_early_accel_scatter.png")
    athlete_df.to_csv(args.output_dir / "athlete_level_analysis.csv", index=False)
    coverage_report.to_csv(args.output_dir / "athlete_level_coverage_report.csv", index=False)
    write_report(args.output_dir, athlete_df, corr_summary, coverage_report)

    if args.self_check:
        run_self_check(athlete_df, args.output_dir)
        return

    print(f"Wrote athlete-level analysis to {args.output_dir / 'athlete_level_analysis.csv'}")
    print(f"Wrote coverage report to {args.output_dir / 'athlete_level_coverage_report.csv'}")
    print(f"Wrote position correlations to {args.output_dir / 'athlete_level_position_corr.csv'}")
    print(f"Wrote scatter plot to {args.output_dir / 'athlete_level_early_accel_scatter.png'}")
    print(f"Wrote short report to {args.output_dir / 'athlete_level_analysis_report.txt'}")


if __name__ == "__main__":
    main()
