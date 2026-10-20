import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA_DIR = Path(r"D:\NFL\data")
OUTPUT_DIR = Path(r"D:\NFL\outputs")
POSITION_GROUPS = ["DB", "WR"]
MOVEMENT_FEATURES = [
    "early_accel_1s_median",
    "max_speed_median",
    "time_to_90pct_max_speed_s_median",
    "path_length_yd_median",
    "net_displacement_yd_median",
    "signed_max_decel_median",
]
CONVENTIONAL_BASELINE = [
    "combine_height",
    "combine_weight",
    "forty",
    "vertical",
    "broad_jump",
    "ten_yd_split",
    "short_shuttle",
    "bench_reps",
]


def load_athlete_table(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    athlete_df = pd.read_csv(output_dir / "athlete_level_analysis.csv", low_memory=False)
    combine = pd.read_csv(data_dir / "combine_results.csv", low_memory=False)
    needed = [
        "draft_year",
        "nfl_id",
        "combine_position",
        "combine_height",
        "combine_weight",
        "forty",
        "vertical",
        "broad_jump",
        "ten_yd_split",
        "short_shuttle",
        "bench_reps",
    ]
    meta = combine[needed].drop_duplicates()
    athlete_df = athlete_df.merge(meta, on=["draft_year", "nfl_id"], how="left", suffixes=("", "_combine"))
    athlete_df = athlete_df[athlete_df["combine_position"].isin(POSITION_GROUPS)].copy()
    for c in CONVENTIONAL_BASELINE:
        athlete_df[c] = pd.to_numeric(athlete_df[c], errors="coerce")
    return athlete_df.reset_index(drop=True)


def fit_feature_residual_model(df: pd.DataFrame, feature_name: str):
    X = df[CONVENTIONAL_BASELINE].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[feature_name], errors="coerce")
    valid = y.notna() & X.notna().all(axis=1)
    if valid.sum() < 10:
        return None, None, None, None
    Xv = X.loc[valid].to_numpy(dtype=float)
    yv = y.loc[valid].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(Xv)), Xv])
    beta, *_ = np.linalg.lstsq(design, yv, rcond=None)
    pred = design @ beta
    residual = yv - pred
    z = (residual - residual.mean()) / residual.std(ddof=1) if residual.std(ddof=1) > 0 else np.zeros_like(residual)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    out.loc[valid] = z
    return beta, valid, out, pred


def build_position_profiles(athlete_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    profile_rows = []
    feature_summary = []
    split_checks = []

    for pos in POSITION_GROUPS:
        pos_df = athlete_df[athlete_df["combine_position"] == pos].copy()
        if len(pos_df) < 20:
            continue

        for feature in MOVEMENT_FEATURES:
            beta, valid_mask, resid_z, _ = fit_feature_residual_model(pos_df, feature)
            if beta is None:
                continue
            pos_df[f"{feature}_resid_z"] = resid_z
            pos_df[f"{feature}_resid_rank"] = resid_z.rank(pct=True)
            athlete_df.loc[pos_df.index, f"{feature}_resid_z"] = resid_z.values
            athlete_df.loc[pos_df.index, f"{feature}_resid_rank"] = pos_df[f"{feature}_resid_rank"].values

            if resid_z.notna().sum() >= 5:
                vals = resid_z.dropna().to_numpy(dtype=float)
                rng = np.random.default_rng(42)
                boots = rng.choice(vals, size=(500, len(vals)), replace=True)
                mean_boot = boots.mean(axis=1)
                feature_summary.append({
                    "position_group": pos,
                    "feature_name": feature,
                    "n_valid": int(vals.size),
                    "mean_residual_z": float(vals.mean()),
                    "ci_low": float(np.percentile(mean_boot, 2.5)),
                    "ci_high": float(np.percentile(mean_boot, 97.5)),
                })

                idx = np.random.RandomState(42).permutation(len(pos_df))
                split = max(5, int(0.8 * len(pos_df)))
                train_idx = idx[:split]
                test_idx = idx[split:]
                train_df = pos_df.iloc[train_idx].copy()
                test_df = pos_df.iloc[test_idx].copy()
                beta_train, _, _, _ = fit_feature_residual_model(train_df, feature)
                if beta_train is not None:
                    x_test = test_df[CONVENTIONAL_BASELINE].apply(pd.to_numeric, errors="coerce")
                    y_test = pd.to_numeric(test_df[feature], errors="coerce")
                    valid_test = y_test.notna() & x_test.notna().all(axis=1)
                    if valid_test.sum() >= 5:
                        Xv = x_test.loc[valid_test].to_numpy(dtype=float)
                        yv = y_test.loc[valid_test].to_numpy(dtype=float)
                        design = np.column_stack([np.ones(len(Xv)), Xv])
                        pred = design @ beta_train
                        test_resid = yv - pred
                        full_test = pos_df.loc[test_df.index[valid_test], f"{feature}_resid_z"]
                        if full_test.notna().sum() >= 5 and test_resid.size >= 5:
                            rho, _ = spearmanr(test_resid, full_test.to_numpy(dtype=float))
                            split_checks.append({
                                "position_group": pos,
                                "feature_name": feature,
                                "n_holdout": int(valid_test.sum()),
                                "rank_rho": float(rho),
                            })

        movement_cols = [f"{feature}_resid_z" for feature in MOVEMENT_FEATURES if f"{feature}_resid_z" in pos_df.columns]
        if movement_cols:
            pos_df["movement_profile_score"] = pos_df[movement_cols].mean(axis=1)
            pos_df["movement_profile_rank"] = pos_df["movement_profile_score"].rank(pct=True)
            athlete_df.loc[pos_df.index, "movement_profile_score"] = pos_df["movement_profile_score"].values
            athlete_df.loc[pos_df.index, "movement_profile_rank"] = pos_df["movement_profile_rank"].values

            top = pos_df.nlargest(3, "movement_profile_score")[[
                "draft_year", "nfl_id", "combine_position", "forty", "vertical", "broad_jump", "combine_weight", "combine_height", "movement_profile_score"
            ] + movement_cols].copy()
            bottom = pos_df.nsmallest(3, "movement_profile_score")[[
                "draft_year", "nfl_id", "combine_position", "forty", "vertical", "broad_jump", "combine_weight", "combine_height", "movement_profile_score"
            ] + movement_cols].copy()
            for _, row in top.iterrows():
                profile_rows.append({"candidate_group": "strong", "position_group": pos, **row.to_dict()})
            for _, row in bottom.iterrows():
                profile_rows.append({"candidate_group": "weak", "position_group": pos, **row.to_dict()})

    profile_df = pd.DataFrame(profile_rows)
    summary_df = pd.DataFrame(feature_summary)
    split_check_df = pd.DataFrame(split_checks)
    return athlete_df, profile_df, {"feature_summary": summary_df, "split_checks": split_check_df}


def plot_profiles(athlete_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = athlete_df[athlete_df["combine_position"].isin(POSITION_GROUPS)].copy()
    residual_col = "early_accel_1s_resid_z" if "early_accel_1s_resid_z" in plot_df.columns else "early_accel_1s_median"
    plot_df = plot_df.dropna(subset=["forty", residual_col]).copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = {"DB": "#1f77b4", "WR": "#ff7f0e"}
    for ax, pos in zip(axes, POSITION_GROUPS):
        subset = plot_df[plot_df["combine_position"] == pos].copy()
        y_values = subset[residual_col] if residual_col.endswith("_resid_z") else subset[residual_col]
        ax.scatter(subset["forty"], y_values, s=30, alpha=0.8, color=colors[pos])
        if residual_col.endswith("_resid_z"):
            ax.axhline(0.0, color="black", lw=1, alpha=0.5)
        ax.set_title(f"{pos} (n={len(subset)})")
        ax.set_xlabel("40-yard time (s)")
        ax.set_ylabel("Early acceleration residual z" if residual_col.endswith("_resid_z") else "Early acceleration median") if ax == axes[0] else None
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_report(output_dir: Path, athlete_df: pd.DataFrame, profile_df: pd.DataFrame, summary: dict) -> None:
    feature_summary = summary["feature_summary"]
    split_checks = summary["split_checks"]
    lines = []
    lines.append("Phase 13 — Position-specific movement profiles")
    lines.append("===========================================")
    lines.append("Verified feature support:\n")
    lines.append(f"- WR athletes retained: {int((athlete_df['combine_position']=='WR').sum())}")
    lines.append(f"- DB athletes retained: {int((athlete_df['combine_position']=='DB').sum())}")
    lines.append(f"- Movement features used: {', '.join(MOVEMENT_FEATURES)}")
    lines.append("- Excluded: three-cone cut metrics, because real-data coverage remains too limited for a defensible position-wise descriptive analysis.")
    lines.append("- Conventional baseline: combine height, weight, 40-yard time, vertical, broad jump, ten-yard split, short shuttle, bench reps.")
    lines.append("- Draft capital was not used as an athletic measurement.")
    lines.append("")
    lines.append("Selected profile summary:")
    if profile_df.empty:
        lines.append("- No candidate profiles were produced because the sample is too sparse for the selected positions.")
    else:
        strong = profile_df[profile_df["candidate_group"] == "strong"].head(3)
        weak = profile_df[profile_df["candidate_group"] == "weak"].head(3)
        lines.append("- Strongest movement candidates (relative to conventional baseline):")
        for _, row in strong.iterrows():
            lines.append(f"  * {row['combine_position']} | nfl_id={int(row['nfl_id'])} | draft_year={int(row['draft_year'])} | movement_profile_score={row['movement_profile_score']:.3f}")
        lines.append("- Weakest movement candidates (relative to conventional baseline):")
        for _, row in weak.iterrows():
            lines.append(f"  * {row['combine_position']} | nfl_id={int(row['nfl_id'])} | draft_year={int(row['draft_year'])} | movement_profile_score={row['movement_profile_score']:.3f}")
    lines.append("")
    lines.append("Uncertainty summary (mean residual z across valid athletes, bootstrap CI):")
    for _, row in feature_summary.sort_values(["position_group", "feature_name"]).iterrows():
        lines.append(f"- {row['position_group']} {row['feature_name']}: mean={row['mean_residual_z']:.3f} 95% CI [{row['ci_low']:.3f}, {row['ci_high']:.3f}]")
    if not split_checks.empty:
        lines.append("")
        lines.append("Held-out robustness check (Spearman rho between residual ranking on 80% fit and the full-sample ranking for the same holdout athletes):")
        for _, row in split_checks.sort_values(["position_group", "feature_name"]).iterrows():
            lines.append(f"- {row['position_group']} {row['feature_name']}: rho={row['rank_rho']:.3f} (n={int(row['n_holdout'])})")
    lines.append("")
    lines.append("Limitations:")
    lines.append("- This is a descriptive comparison, not a prediction study.")
    lines.append("- There is no valid NFL outcome target in the current minimal analysis; no claim is made about future performance or hidden star status.")
    lines.append("- The analysis intentionally avoids arbitrary thresholding and only flags the most extreme 10% of the within-position residual distribution as candidate profiles.")
    lines.append("- The DB backpedal transition feature remains low coverage and was not used in the primary position-wise comparison.")

    report_path = output_dir / "phase13_position_profiles_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def self_check(athlete_df: pd.DataFrame) -> None:
    assert set(POSITION_GROUPS).issubset(set(athlete_df["combine_position"].dropna().unique())), "Missing WR/DB athletes"
    assert (athlete_df["combine_position"] == "WR").sum() >= 20, "Too few WR athletes for a position profile analysis"
    assert (athlete_df["combine_position"] == "DB").sum() >= 20, "Too few DB athletes for a position profile analysis"
    for feature in MOVEMENT_FEATURES:
        assert feature in athlete_df.columns, f"Missing feature {feature}"
    print("SELF_CHECK_OK")
    print("WR_athletes", int((athlete_df["combine_position"] == "WR").sum()))
    print("DB_athletes", int((athlete_df["combine_position"] == "DB").sum()))
    print("movement_features", MOVEMENT_FEATURES)


def main():
    parser = argparse.ArgumentParser(description="Phase 13: position-specific movement profiles")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    athlete_df = load_athlete_table(args.data_dir, args.output_dir)
    if args.self_check:
        self_check(athlete_df)
        return

    athlete_df, profile_df, summary = build_position_profiles(athlete_df)
    athlete_df = athlete_df.sort_values(["combine_position", "draft_year", "nfl_id"]).reset_index(drop=True)
    athlete_df.to_csv(args.output_dir / "phase13_position_profiles.csv", index=False)
    plot_profiles(athlete_df, args.output_dir / "phase13_position_profiles.png")
    write_report(args.output_dir, athlete_df, profile_df, summary)
    summary["feature_summary"].to_csv(args.output_dir / "phase13_position_feature_summary.csv", index=False)
    if not summary["split_checks"].empty:
        summary["split_checks"].to_csv(args.output_dir / "phase13_position_split_checks.csv", index=False)

    print(f"Wrote tidy athlete table to {args.output_dir / 'phase13_position_profiles.csv'}")
    print(f"Wrote report to {args.output_dir / 'phase13_position_profiles_report.txt'}")
    print(f"Wrote plot to {args.output_dir / 'phase13_position_profiles.png'}")
    print(f"Candidate strong profiles: {len(profile_df[profile_df['candidate_group']=='strong'])}")
    print(f"Candidate weak profiles: {len(profile_df[profile_df['candidate_group']=='weak'])}")


if __name__ == "__main__":
    main()
