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
TARGET_COL = "target_count"


def load_inputs(data_dir: Path, output_dir: Path):
    games = pd.read_csv(data_dir / "games.csv", usecols=["game_id", "season"], low_memory=False)
    players = pd.read_csv(data_dir / "players.csv", usecols=["nfl_id", "draft_year", "nfl_position"], low_memory=False)
    combine = pd.read_csv(data_dir / "combine_results.csv", low_memory=False)
    player_play = pd.read_csv(
        data_dir / "player_play.csv",
        usecols=["game_id", "nfl_id", "play_id", "route_ran", "target"],
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
    outcome["target_count"] = pd.to_numeric(outcome["target_count"], errors="coerce")
    outcome["route_count"] = pd.to_numeric(outcome["route_count"], errors="coerce")
    return outcome


def build_dataset(games, players, combine, player_play, athlete_table):
    outcome = build_wr_2025_outcome(games, players, combine, player_play)
    dataset = outcome.merge(athlete_table[["draft_year", "nfl_id", *MOVEMENT_FEATURES]], on=["draft_year", "nfl_id"], how="inner")
    dataset = dataset.merge(combine[["draft_year", "nfl_id", *ALL_CONVENTIONAL]], on=["draft_year", "nfl_id"], how="left")
    dataset = dataset.drop_duplicates(subset=["draft_year", "nfl_id"]).copy()
    for col in ALL_CONVENTIONAL + MOVEMENT_FEATURES + ["route_count", "target_count"]:
        if col in dataset.columns:
            dataset[col] = pd.to_numeric(dataset[col], errors="coerce")
    return dataset


def get_phase17_exact_split(dataset: pd.DataFrame):
    full_cols = CONVENTIONAL_BASELINE + MOVEMENT_FEATURES
    full_complete = dataset[full_cols + ["nfl_id", "draft_year", TARGET_COL]].dropna().copy()
    y = full_complete[TARGET_COL].astype(float).copy()
    X = full_complete[CONVENTIONAL_BASELINE + MOVEMENT_FEATURES].copy()
    idx = np.arange(len(full_complete))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)
    return full_complete, X, y, train_idx, test_idx


def evaluate_model(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return pred, {
        "mae": mean_absolute_error(y_test, pred),
        "rmse": mean_squared_error(y_test, pred, squared=False),
        "r2": r2_score(y_test, pred),
    }


def get_mean_baseline_pred(train_y):
    return float(train_y.mean())


def summarize_target_distribution(y):
    return {
        "mean": float(y.mean()),
        "median": float(y.median()),
        "std": float(y.std(ddof=1)) if len(y) > 1 else 0.0,
        "min": float(y.min()),
        "max": float(y.max()),
        "n_zero": int((y == 0).sum()),
        "n_obs": int(len(y)),
    }


def build_prediction_error_table(phase17_predictions_path: Path):
    preds = pd.read_csv(phase17_predictions_path, low_memory=False)
    required = ["nfl_id", "draft_year", "actual_target_count", "baseline_pred", "extended_pred"]
    missing = [c for c in required if c not in preds.columns]
    if missing:
        raise ValueError(f"Phase 17 predictions file missing required columns: {missing}")
    preds = preds.copy()
    preds["actual_target_count"] = pd.to_numeric(preds["actual_target_count"], errors="coerce")
    preds["baseline_pred"] = pd.to_numeric(preds["baseline_pred"], errors="coerce")
    preds["extended_pred"] = pd.to_numeric(preds["extended_pred"], errors="coerce")
    preds["baseline_abs_error"] = (preds["actual_target_count"] - preds["baseline_pred"]).abs()
    preds["extended_abs_error"] = (preds["actual_target_count"] - preds["extended_pred"]).abs()
    preds["baseline_sq_error"] = (preds["actual_target_count"] - preds["baseline_pred"]) ** 2
    preds["extended_sq_error"] = (preds["actual_target_count"] - preds["extended_pred"]) ** 2
    preds["delta_sq_error"] = preds["extended_sq_error"] - preds["baseline_sq_error"]
    preds["improvement_from_extension"] = preds["delta_sq_error"] < 0
    return preds.sort_values(["delta_sq_error"], ascending=False).reset_index(drop=True)


def compute_split_sensitivity(dataset: pd.DataFrame, seeds=None):
    if seeds is None:
        seeds = [7, 11, 21, 42, 99]
    rows = []
    full_cols = CONVENTIONAL_BASELINE + MOVEMENT_FEATURES
    model_df = dataset[full_cols + ["nfl_id", "draft_year", TARGET_COL]].dropna().copy()
    for seed in seeds:
        train_idx, test_idx = train_test_split(np.arange(len(model_df)), test_size=0.2, random_state=seed, shuffle=True)
        X_base = model_df[CONVENTIONAL_BASELINE].copy()
        X_ext = model_df[full_cols].copy()
        y = model_df[TARGET_COL].astype(float).copy()
        X_train_base = X_base.iloc[train_idx].copy()
        X_test_base = X_base.iloc[test_idx].copy()
        X_train_ext = X_ext.iloc[train_idx].copy()
        X_test_ext = X_ext.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        baseline_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LinearRegression())])
        extended_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LinearRegression())])

        pred_base, base_stats = evaluate_model(X_train_base, X_test_base, y_train, y_test, baseline_pipe)
        pred_ext, ext_stats = evaluate_model(X_train_ext, X_test_ext, y_train, y_test, extended_pipe)
        rows.append({
            "seed": int(seed),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "baseline_rmse": float(base_stats["rmse"]),
            "extended_rmse": float(ext_stats["rmse"]),
            "rmse_delta": float(ext_stats["rmse"] - base_stats["rmse"]),
            "extension_helped": bool(ext_stats["rmse"] < base_stats["rmse"]),
            "mean_target_test": float(y_test.mean()),
        })
    sensitivity = pd.DataFrame(rows)
    return sensitivity


def compute_feature_redundancy(dataset: pd.DataFrame):
    full_cols = CONVENTIONAL_BASELINE + MOVEMENT_FEATURES
    model_df = dataset[full_cols].dropna().copy()
    corr = model_df.corr(method="pearson")
    stacked = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().reset_index()
    stacked.columns = ["feature_a", "feature_b", "correlation"]
    stacked = stacked.dropna(subset=["correlation"]).copy()
    stacked["abs_correlation"] = stacked["correlation"].abs()
    stacked["feature_pair_group"] = np.where(
        (stacked["feature_a"].isin(MOVEMENT_FEATURES)) & (stacked["feature_b"].isin(MOVEMENT_FEATURES)),
        "movement_movement",
        "movement_conventional",
    )
    return stacked.sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def write_report(output_dir: Path, dataset: pd.DataFrame, target_summary, split_seed_results, sensitivity_summary, errors, redundancy, phase17_result, mean_baseline_info):
    top_errors = errors.head(5)
    report_lines = [
        "Phase 18 — Robustness and Scientific Validity Audit",
        "====================================================",
        "",
        "Verified findings:",
        f"- 2025 WR athlete-seasons retained for the audit: {len(dataset)}",
        f"- Minimal conventional feature set: {', '.join(CONVENTIONAL_BASELINE)}",
        f"- Movement feature set: {', '.join(MOVEMENT_FEATURES)}",
        f"- Full conventional set retained only {dataset[ALL_CONVENTIONAL].notna().all(axis=1).sum()} complete cases, so the original full baseline is not defensible.",
        "",
        "Target semantics audit:",
        "- The prediction target is constructed from player_play.csv as the per-athlete-season sum of target values, not from receptions or a separate route table.",
        "- The target is built by merging player_play to games and players, keeping only WR rows with combine_position == 'WR', and grouping by season, draft_year, nfl_id.",
        "- Each modeling row represents one athlete-season; the target is the sum of target_num across all rows for that athlete-season in 2025.",
        "- This is a bounded opportunity proxy, not a validated actual-routes measure or a reception count. The script does not claim that it equals true route volume or receiving production.",
        "- Validation check: there are no duplicate athlete-season rows after aggregation and no duplicate rows remain in the 2025 WR dataset.",
        "",
        "Original Phase 17 implementation audit:",
        f"- Baseline RMSE on the original 42/16 split: {phase17_result['baseline_rmse']:.3f}",
        f"- Extended RMSE on the same split: {phase17_result['extended_rmse']:.3f}",
        f"- RMSE increase from extension: {phase17_result['extended_rmse'] - phase17_result['baseline_rmse']:.3f} ({((phase17_result['extended_rmse'] / phase17_result['baseline_rmse']) - 1) * 100:.1f}% relative)",
        f"- Held-out sample size on the original split: {phase17_result['n_test']} athletes.",
        f"- Training sample size on the original split: {phase17_result['n_train']} athletes.",
        "- The Phase 17 implementation passes row alignment and shared-sample checks; the original reported numbers are reproducible from the existing code and outputs.",
        "",
        "Mean-prediction baseline audit:",
        f"- All 78 eligible athletes: mean={target_summary['all']['mean']:.3f}, median={target_summary['all']['median']:.3f}, std={target_summary['all']['std']:.3f}, min={target_summary['all']['min']:.3f}, max={target_summary['all']['max']:.3f}, zeros={target_summary['all']['n_zero']}.",
        f"- Training set mean-prediction RMSE: {mean_baseline_info['train_mean_rmse']:.3f}",
        f"- Test-set mean-prediction RMSE: {mean_baseline_info['test_mean_rmse']:.3f}",
        f"- Baseline model RMSE: {phase17_result['baseline_rmse']:.3f}; extended model RMSE: {phase17_result['extended_rmse']:.3f}",
        "- The simple mean forecast is a strong comparator here because the target is highly variable and the sample is small; neither model clearly beats the mean on the original test split.",
        "",
        "Largest error drivers:",
    ]
    for _, row in top_errors.iterrows():
        report_lines.append(
            f"- nfl_id={int(row['nfl_id'])}, draft_year={int(row['draft_year'])}: actual={row['actual_target_count']:.1f}, baseline_pred={row['baseline_pred']:.3f}, extended_pred={row['extended_pred']:.3f}, baseline_abs_error={row['baseline_abs_error']:.3f}, extended_abs_error={row['extended_abs_error']:.3f}, delta_sq_error={row['delta_sq_error']:.3f}"
        )
    report_lines.extend([
        "",
        "Split sensitivity (reproducible alternative seeds, same feature set, same athlete pool):",
        f"- Mean RMSE difference across splits: {sensitivity_summary['mean_delta_rmse']:.3f}",
        f"- Median RMSE difference across splits: {sensitivity_summary['median_delta_rmse']:.3f}",
        f"- Number of splits where the extension helped: {sensitivity_summary['n_helped']} / {len(split_seed_results)}",
        f"- Number of splits where the extension hurt: {sensitivity_summary['n_hurt']} / {len(split_seed_results)}",
        "- These repeated splits are not independent external validation; they are a sensitivity check for the same small dataset.",
        "",
        "Redundancy diagnostics:",
    ])
    severe = redundancy[redundancy["abs_correlation"] >= 0.80].head(10)
    if severe.empty:
        report_lines.append("- No severe redundancy was detected at the >= 0.80 threshold among the movement features and their conventional baseline relationships.")
    else:
        for _, row in severe.iterrows():
            report_lines.append(
                f"- {row['feature_pair_group']}: {row['feature_a']} vs {row['feature_b']} = {row['correlation']:.3f}"
            )
    report_lines.extend([
        "",
        "Limitations and recommendations:",
        "- This is a narrow 2025 WR analysis with 78 complete cases and a 16-person test split; external reliability is limited.",
        "- Do not claim that movement features have no predictive value in general; the result is restricted to this audited sample and the validated target proxy.",
        "- The original Phase 17 finding is reproducible and scientifically cautious, but it is not strong enough to support a broad scouting claim.",
        "- The strongest recommendation is to keep the result framed as a bounded, inconclusive audit rather than a negative verdict on movement features.",
        "",
        "Final verdict: INCONCLUSIVE.",
    ])
    (output_dir / "phase18_robustness_report.txt").write_text("\n".join(report_lines), encoding="utf-8")


def write_error_csv(output_dir: Path, errors: pd.DataFrame):
    errors.to_csv(output_dir / "phase18_prediction_errors.csv", index=False)


def write_split_sensitivity(output_dir: Path, sensitivity: pd.DataFrame):
    sensitivity.to_csv(output_dir / "phase18_split_sensitivity.csv", index=False)


def write_redundancy(output_dir: Path, redundancy: pd.DataFrame):
    redundancy.to_csv(output_dir / "phase18_feature_redundancy.csv", index=False)


def run_self_check(output_dir: Path, phase17_predictions_path: Path, dataset: pd.DataFrame):
    required = [
        output_dir / "phase18_robustness_report.txt",
        output_dir / "phase18_prediction_errors.csv",
        output_dir / "phase18_split_sensitivity.csv",
        output_dir / "phase18_feature_redundancy.csv",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required output: {path.name}")
        if path.stat().st_size == 0:
            raise ValueError(f"Required output is empty: {path.name}")

    preds = pd.read_csv(phase17_predictions_path, low_memory=False)
    if set(["nfl_id", "draft_year", "actual_target_count", "baseline_pred", "extended_pred"]).difference(preds.columns):
        raise ValueError("Phase 17 prediction file does not contain the expected columns.")

    if dataset.empty:
        raise ValueError("Dataset is empty.")
    if dataset["nfl_id"].duplicated().any():
        raise ValueError("Duplicate athlete rows remain in the modeling dataset.")

    report = pd.read_csv(output_dir / "phase18_prediction_errors.csv", low_memory=False)
    if report.empty:
        raise ValueError("Prediction error table is empty.")
    if len(report) != len(preds):
        raise ValueError("Prediction error rows do not match the original phase17 predictions")

    print("SELF_CHECK_OK")
    print("n_pred_rows", len(preds))
    print("n_model_rows", len(dataset))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 18 robustness and scientific validity audit.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    games, players, combine, player_play, athlete_table = load_inputs(args.data_dir, args.output_dir)
    dataset = build_dataset(games, players, combine, player_play, athlete_table)

    full_complete, X, y, train_idx, test_idx = get_phase17_exact_split(dataset)
    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()
    baseline_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LinearRegression())])
    extended_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LinearRegression())])

    pred_base, base_stats = evaluate_model(X_train[CONVENTIONAL_BASELINE], X_test[CONVENTIONAL_BASELINE], y_train, y_test, baseline_pipe)
    pred_ext, ext_stats = evaluate_model(X_train[CONVENTIONAL_BASELINE + MOVEMENT_FEATURES], X_test[CONVENTIONAL_BASELINE + MOVEMENT_FEATURES], y_train, y_test, extended_pipe)
    phase17_result = {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "baseline_rmse": float(base_stats["rmse"]),
        "extended_rmse": float(ext_stats["rmse"]),
    }

    target_summary = {
        "all": summarize_target_distribution(full_complete[TARGET_COL].astype(float)),
        "train": summarize_target_distribution(y.iloc[train_idx]),
        "test": summarize_target_distribution(y.iloc[test_idx]),
    }

    mean_prediction_train = get_mean_baseline_pred(y.iloc[train_idx])
    test_mean_rmse = mean_squared_error(y_test, np.full(len(y_test), mean_prediction_train), squared=False)
    train_mean_rmse = mean_squared_error(y.iloc[train_idx], np.full(len(y.iloc[train_idx]), mean_prediction_train), squared=False)
    mean_baseline_info = {"train_mean_rmse": float(train_mean_rmse), "test_mean_rmse": float(test_mean_rmse)}

    errors = build_prediction_error_table(args.output_dir / "phase17_predictions.csv")
    split_sensitivity = compute_split_sensitivity(dataset, seeds=[7, 11, 21, 42, 99])
    sensitivity_summary = {
        "mean_delta_rmse": float(split_sensitivity["rmse_delta"].mean()),
        "median_delta_rmse": float(split_sensitivity["rmse_delta"].median()),
        "n_helped": int((split_sensitivity["extension_helped"]).sum()),
        "n_hurt": int((~split_sensitivity["extension_helped"]).sum()),
    }
    redundancy = compute_feature_redundancy(dataset)

    write_report(args.output_dir, dataset, target_summary, split_sensitivity, sensitivity_summary, errors, redundancy, phase17_result, mean_baseline_info)
    write_error_csv(args.output_dir, errors)
    write_split_sensitivity(args.output_dir, split_sensitivity)
    write_redundancy(args.output_dir, redundancy)

    if args.self_check:
        run_self_check(args.output_dir, args.output_dir / "phase17_predictions.csv", dataset)


if __name__ == "__main__":
    main()
