import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DEFAULT_DATA_DIR = Path(r"D:\NFL\data")
DEFAULT_OUTPUT_DIR = Path(r"D:\NFL\outputs")

CONVENTIONAL_BASELINE = [
    "combine_height",
    "combine_weight",
    "forty",
    "vertical",
    "broad_jump",
    "ten_yd_split",
]
MOVEMENT_FEATURES = [
    "early_accel_1s_median",
    "max_speed_median",
    "time_to_90pct_max_speed_s_median",
    "path_length_yd_median",
    "net_displacement_yd_median",
    "signed_max_decel_median",
]
ALL_CONVENTIONAL = [
    "combine_height",
    "combine_weight",
    "forty",
    "vertical",
    "broad_jump",
    "ten_yd_split",
    "three_cone",
    "short_shuttle",
    "bench_reps",
]

MIN_TOTAL_COMPLETE_CASES = 60
MIN_TEST_CASES = 15


def load_inputs(data_dir: Path, output_dir: Path):
    games = pd.read_csv(data_dir / "games.csv", usecols=["game_id", "season"], low_memory=False)
    players = pd.read_csv(data_dir / "players.csv", usecols=["nfl_id", "draft_year", "nfl_position"], low_memory=False)
    combine = pd.read_csv(data_dir / "combine_results.csv", low_memory=False)
    player_play = pd.read_csv(
        data_dir / "player_play.csv",
        usecols=["game_id", "nfl_id", "route_ran", "target"],
        low_memory=False,
    )
    athlete_table = pd.read_csv(output_dir / "athlete_level_analysis.csv", low_memory=False)
    return games, players, combine, player_play, athlete_table


def build_wr_2025_outcome(games, players, combine, player_play):
    season_play = player_play.merge(games[["game_id", "season"]], on="game_id", how="left")
    season_play = season_play.merge(players[["nfl_id", "draft_year"]], on="nfl_id", how="left")
    season_play = season_play.merge(combine[["nfl_id", "draft_year", "combine_position"]], on=["nfl_id", "draft_year"], how="left")
    season_play["season"] = pd.to_numeric(season_play["season"], errors="coerce")
    season_play["route_flag"] = season_play["route_ran"].notna().astype(int)
    season_play["target_num"] = pd.to_numeric(season_play["target"], errors="coerce").fillna(0).astype(float)

    wr = season_play[season_play["combine_position"] == "WR"].copy()
    outcome = (
        wr.groupby(["season", "draft_year", "nfl_id"], dropna=False)
        .agg(route_count=("route_flag", "sum"), target_count=("target_num", "sum"))
        .reset_index()
    )
    outcome = outcome[outcome["season"] == 2025].copy()
    return outcome


def build_dataset(games, players, combine, player_play, athlete_table):
    outcome = build_wr_2025_outcome(games, players, combine, player_play)
    dataset = outcome.merge(
        athlete_table[["draft_year", "nfl_id", *MOVEMENT_FEATURES]],
        on=["draft_year", "nfl_id"],
        how="inner",
    )
    dataset = dataset.merge(
        combine[["draft_year", "nfl_id", *ALL_CONVENTIONAL]],
        on=["draft_year", "nfl_id"],
        how="left",
    )
    dataset = dataset.drop_duplicates(subset=["draft_year", "nfl_id"]).copy()
    for col in ALL_CONVENTIONAL + MOVEMENT_FEATURES + ["route_count", "target_count"]:
        if col in dataset.columns:
            dataset[col] = pd.to_numeric(dataset[col], errors="coerce")
    return dataset


def summarize_coverage(dataset: pd.DataFrame):
    rows = []
    for feature in ALL_CONVENTIONAL + MOVEMENT_FEATURES:
        if feature not in dataset.columns:
            continue
        series = dataset[feature]
        rows.append(
            {
                "feature_group": "conventional" if feature in ALL_CONVENTIONAL else "movement",
                "feature": feature,
                "n_non_missing": int(series.notna().sum()),
                "n_missing": int(series.isna().sum()),
                "coverage_pct": float(series.notna().mean() * 100.0),
                "selected_for_minimal_baseline": feature in CONVENTIONAL_BASELINE,
            }
        )
    coverage = pd.DataFrame(rows).sort_values(["feature_group", "feature"]).reset_index(drop=True)

    candidate_sets = {
        "all_conventional": ALL_CONVENTIONAL,
        "minimal_conventional": CONVENTIONAL_BASELINE,
        "minimal_extended": CONVENTIONAL_BASELINE + MOVEMENT_FEATURES,
    }
    candidate_rows = []
    for name, cols in candidate_sets.items():
        present = [c for c in cols if c in dataset.columns]
        complete = dataset[present].notna().all(axis=1).sum() if present else 0
        candidate_rows.append({
            "candidate_set": name,
            "n_features": len(present),
            "n_complete_cases": int(complete),
        })
    candidates = pd.DataFrame(candidate_rows)
    return coverage, candidates


def fit_model_and_compare(dataset: pd.DataFrame):
    baseline_cols = CONVENTIONAL_BASELINE
    movement_cols = MOVEMENT_FEATURES
    target_col = "target_count"
    full_cols = baseline_cols + movement_cols

    complete = dataset[baseline_cols + [target_col]].dropna().copy()
    full_complete = dataset[full_cols + [target_col]].dropna().copy()
    if len(full_complete) < MIN_TOTAL_COMPLETE_CASES:
        return None, f"Not enough complete data for a defensible comparison; total complete cases={len(full_complete)} < {MIN_TOTAL_COMPLETE_CASES}."

    model_df = dataset.loc[full_complete.index, ["nfl_id", "draft_year", *full_cols, target_col]].copy()
    ids = model_df[["nfl_id", "draft_year"]].copy()
    X_base = model_df[baseline_cols].copy()
    X_ext = model_df[full_cols].copy()
    y = model_df[target_col].astype(float).copy()

    train_idx, test_idx = train_test_split(np.arange(len(model_df)), test_size=0.2, random_state=42, shuffle=True)
    if len(test_idx) < MIN_TEST_CASES:
        return None, f"Test split too small; n_test={len(test_idx)} < {MIN_TEST_CASES}."

    X_train_base = X_base.iloc[train_idx].copy()
    X_test_base = X_base.iloc[test_idx].copy()
    X_train_ext = X_ext.iloc[train_idx].copy()
    X_test_ext = X_ext.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()
    ids_test = ids.iloc[test_idx].copy()

    baseline_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LinearRegression()),
    ])
    baseline_pipe.fit(X_train_base, y_train)
    pred_base = baseline_pipe.predict(X_test_base)

    ext_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LinearRegression()),
    ])
    ext_pipe.fit(X_train_ext, y_train)
    pred_ext = ext_pipe.predict(X_test_ext)

    result = {
        "n_total_complete": int(len(model_df)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "target_mean": float(y_test.mean()),
        "baseline_mae": float(mean_absolute_error(y_test, pred_base)),
        "baseline_rmse": float(mean_squared_error(y_test, pred_base, squared=False)),
        "baseline_r2": float(r2_score(y_test, pred_base)),
        "extended_mae": float(mean_absolute_error(y_test, pred_ext)),
        "extended_rmse": float(mean_squared_error(y_test, pred_ext, squared=False)),
        "extended_r2": float(r2_score(y_test, pred_ext)),
        "leakage_check": bool(ids_test[["nfl_id"]].drop_duplicates().shape[0] == ids_test[["nfl_id"]].shape[0]),
        "predictions": pd.DataFrame(
            {
                "nfl_id": ids_test["nfl_id"].to_numpy(),
                "draft_year": ids_test["draft_year"].to_numpy(),
                "actual_target_count": y_test.to_numpy(),
                "baseline_pred": pred_base,
                "extended_pred": pred_ext,
            }
        ),
    }
    return result, None


def write_missingness_report(output_dir: Path, dataset: pd.DataFrame, coverage: pd.DataFrame, candidates: pd.DataFrame, result):
    lines = [
        "Phase 17 — Recover a Defensible Combine-to-NFL Experiment",
        "========================================================",
        "",
        "Feasibility gate:",
        "- Minimum rule: at least 60 complete-case athletes in the minimal conventional baseline and at least 15 test-set athletes before a comparison is considered stable enough to report.",
        "- Rationale: the prior Phase 16 failure was caused by dropping most of the sample when using the full conventional baseline. This recovery keeps the model small, uses the same athletes in both models, and avoids a test split that is too small to support a meaningful comparison.",
        "",
        "2025 WR cohort audit:",
        f"- WR athlete-season records: {len(dataset)}",
    ]

    lines.extend([
        "- Feature coverage by metric:",
    ])
    for _, row in coverage.iterrows():
        lines.append(
            f"  * {row['feature_group']} {row['feature']}: {int(row['n_non_missing'])}/{int(row['n_non_missing'] + row['n_missing'])} non-missing ({row['coverage_pct']:.1f}%)"
        )

    lines.extend([
        "",
        "Candidate complete-case sets:",
    ])
    for _, row in candidates.iterrows():
        lines.append(
            f"  * {row['candidate_set']}: {int(row['n_complete_cases'])} complete cases across {int(row['n_features'])} features"
        )

    if result is not None:
        lines.extend([
            "",
            "Feasibility status:",
            f"- PASS: total complete cases={result['n_total_complete']} >= {MIN_TOTAL_COMPLETE_CASES}; test cases={result['n_test']} >= {MIN_TEST_CASES}",
            f"- Baseline MAE={result['baseline_mae']:.3f}, RMSE={result['baseline_rmse']:.3f}, R2={result['baseline_r2']:.3f}",
            f"- Extended MAE={result['extended_mae']:.3f}, RMSE={result['extended_rmse']:.3f}, R2={result['extended_r2']:.3f}",
            "- Interpretation: the minimal comparison is feasible but still small and unstable; it is a bounded, descriptive check rather than proof of causal scouting value.",
            "- Decision: CONDITIONAL GO. A defensible experiment can be reported, but only with the caveat that the sample is limited and the result is not generalizable beyond the 2025 WR cohort.",
        ])
    else:
        lines.extend([
            "",
            "Feasibility status:",
            "- NO-GO: the available sample is too limited for a defensible model comparison under the minimum sample rule.",
            "- The reason is documented above and no model output is reported.",
        ])

    (output_dir / "phase17_missingness_report.txt").write_text("\n".join(lines), encoding="utf-8")


def write_feature_coverage(output_dir: Path, coverage: pd.DataFrame):
    coverage.to_csv(output_dir / "phase17_feature_coverage.csv", index=False)


def run_self_check(output_dir: Path, dataset: pd.DataFrame, coverage: pd.DataFrame, candidates: pd.DataFrame) -> None:
    required = [
        output_dir / "phase17_missingness_report.txt",
        output_dir / "phase17_feature_coverage.csv",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required output: {path.name}")
        if path.stat().st_size == 0:
            raise ValueError(f"Required output is empty: {path.name}")
    if dataset.empty:
        raise ValueError("Validated 2025 WR dataset is empty.")
    if dataset["season"].nunique() != 1:
        raise ValueError("Dataset is not restricted to a single season.")
    if int(dataset["season"].iloc[0]) != 2025:
        raise ValueError("Dataset is not the 2025 WR sample.")
    if coverage.empty:
        raise ValueError("Coverage table is empty.")
    if candidates.empty:
        raise ValueError("Candidate set table is empty.")
    print("SELF_CHECK_OK")
    print("WR_2025_count", len(dataset))
    print("minimal_complete_cases", int(candidates.loc[candidates["candidate_set"] == "minimal_conventional", "n_complete_cases"].iloc[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 17 missingness recovery and minimal comparison audit.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    games, players, combine, player_play, athlete_table = load_inputs(args.data_dir, args.output_dir)
    dataset = build_dataset(games, players, combine, player_play, athlete_table)
    coverage, candidates = summarize_coverage(dataset)
    write_feature_coverage(args.output_dir, coverage)

    result, reason = fit_model_and_compare(dataset)
    write_missingness_report(args.output_dir, dataset, coverage, candidates, result)

    if result is not None:
        model_path = args.output_dir / "phase17_model_comparison_report.txt"
        predictions_path = args.output_dir / "phase17_predictions.csv"
        lines = [
            "Phase 17 — minimal baseline-versus-extended WR comparison",
            "========================================================",
            "",
            f"Total complete-case sample: {result['n_total_complete']}",
            f"Training sample: {result['n_train']}",
            f"Test sample: {result['n_test']}",
            f"Target mean in test split: {result['target_mean']:.3f}",
            "",
            f"Baseline RMSE: {result['baseline_rmse']:.3f}",
            f"Baseline MAE: {result['baseline_mae']:.3f}",
            f"Baseline R2: {result['baseline_r2']:.3f}",
            f"Extended RMSE: {result['extended_rmse']:.3f}",
            f"Extended MAE: {result['extended_mae']:.3f}",
            f"Extended R2: {result['extended_r2']:.3f}",
            "",
            "Decision: CONDITIONAL GO.",
            "- The comparison uses the same 2025 WR athletes, the same target definition, the same train/test split, and the same missing-data handling policy.",
            "- The dataset is small and the result is therefore limited; it should be read as a bounded descriptive model check, not as proof of hidden scouting value.",
            "- This is the smallest defensible comparison available under the current coverage constraints.",
        ]
        model_path.write_text("\n".join(lines), encoding="utf-8")
        result["predictions"].to_csv(predictions_path, index=False)

    if args.self_check:
        run_self_check(args.output_dir, dataset, coverage, candidates)


if __name__ == "__main__":
    main()
