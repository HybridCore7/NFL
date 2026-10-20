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
    "ten_yd_split",
    "forty",
    "vertical",
    "broad_jump",
    "three_cone",
    "short_shuttle",
    "bench_reps",
]
MOVEMENT_FEATURES = [
    "early_accel_1s_median",
    "max_speed_median",
    "time_to_90pct_max_speed_s_median",
    "path_length_yd_median",
    "net_displacement_yd_median",
    "signed_max_decel_median",
]


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


def build_wr_player_season_outcomes(games: pd.DataFrame, players: pd.DataFrame, combine: pd.DataFrame, player_play: pd.DataFrame) -> pd.DataFrame:
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
    outcome["targets_per_route"] = np.where(outcome["route_count"] > 0, outcome["target_count"] / outcome["route_count"], np.nan)
    return outcome


def build_validated_dataset(games, players, combine, player_play, athlete_table):
    outcomes = build_wr_player_season_outcomes(games, players, combine, player_play)
    outcomes = outcomes[outcomes["season"] == 2025].copy()

    validate_table = outcomes.merge(
        athlete_table[["draft_year", "nfl_id", *MOVEMENT_FEATURES]],
        on=["draft_year", "nfl_id"],
        how="inner",
    )
    validate_table = validate_table.merge(
        combine[["draft_year", "nfl_id", *CONVENTIONAL_BASELINE]],
        on=["draft_year", "nfl_id"],
        how="inner",
    )
    validate_table = validate_table.drop_duplicates(subset=["draft_year", "nfl_id"]).copy()
    validate_table["route_count"] = pd.to_numeric(validate_table["route_count"], errors="coerce")
    validate_table["target_count"] = pd.to_numeric(validate_table["target_count"], errors="coerce")
    validate_table["targets_per_route"] = pd.to_numeric(validate_table["targets_per_route"], errors="coerce")
    return validate_table


def fit_and_evaluate(df: pd.DataFrame, target_col: str):
    if df.empty:
        return None

    baseline_cols = CONVENTIONAL_BASELINE
    full_cols = baseline_cols + MOVEMENT_FEATURES
    X = df[baseline_cols].copy()
    X_ext = df[full_cols].copy()
    y = df[target_col].astype(float).copy()
    valid = y.notna() & X.notna().all(axis=1) & X_ext.notna().all(axis=1)
    X = X.loc[valid].copy()
    X_ext = X_ext.loc[valid].copy()
    y = y.loc[valid].copy()
    if len(y) < 10:
        return None

    athlete_ids = df.loc[valid, ["nfl_id", "draft_year"]].copy()
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, athlete_ids, test_size=0.3, random_state=42
    )
    X_train_ext = X_ext.loc[X_train.index]
    X_test_ext = X_ext.loc[X_test.index]
    leakage_ok = set(ids_train["nfl_id"]).isdisjoint(set(ids_test["nfl_id"]))

    baseline_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LinearRegression())])
    baseline_pipe.fit(X_train, y_train)
    pred = baseline_pipe.predict(X_test)
    baseline_metrics = {
        "mae": mean_absolute_error(y_test, pred),
        "rmse": mean_squared_error(y_test, pred, squared=False),
        "r2": r2_score(y_test, pred),
    }

    ext_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LinearRegression())])
    ext_pipe.fit(X_train_ext, y_train)
    pred_ext = ext_pipe.predict(X_test_ext)
    ext_metrics = {
        "mae": mean_absolute_error(y_test, pred_ext),
        "rmse": mean_squared_error(y_test, pred_ext, squared=False),
        "r2": r2_score(y_test, pred_ext),
    }

    return {
        "target": target_col,
        "n_rows": int(len(y)),
        "baseline": baseline_metrics,
        "extended": ext_metrics,
        "leakage_check": leakage_ok,
    }


def write_report(output_dir: Path, dataset: pd.DataFrame, experiment_metrics: dict) -> None:
    lines = [
        "Phase 16 — Outcome validation and incremental predictive value",
        "==============================================================",
        "",
        "Validated sample:",
        f"- WR athletes in the 2025 season: {len(dataset)}",
        f"- Mean route count: {dataset['route_count'].mean():.2f}",
        f"- Mean target count: {dataset['target_count'].mean():.2f}",
        "",
        "Experiment design:",
        "- Baseline uses the conventional Combine metrics only.",
        "- Extended model adds the validated movement features from the athlete-level output.",
        "- The same athlete set and deterministic split are used for both models.",
        "- This remains a narrow WR-2025 check, not a general NFL success model.",
        "",
        "Results:",
    ]

    if not experiment_metrics:
        lines.append("- No valid experiment could be run with the current filtered WR sample.")
    else:
        for outcome_name, result in experiment_metrics.items():
            baseline = result["baseline"]
            extended = result["extended"]
            lines.append(
                f"- Outcome: {outcome_name}; baseline MAE={baseline['mae']:.3f}, RMSE={baseline['rmse']:.3f}, R2={baseline['r2']:.3f}; extended MAE={extended['mae']:.3f}, RMSE={extended['rmse']:.3f}, R2={extended['r2']:.3f}; leakage_check={result['leakage_check']}"
            )
            if extended["rmse"] < baseline["rmse"]:
                lines.append("  * The extended model improved test RMSE in this narrow 2025 WR sample.")
            else:
                lines.append("  * The extended model did not improve test RMSE in this narrow 2025 WR sample.")

    lines.extend([
        "",
        "Decision:",
        "- A single-season WR outcome check is defensible as a bounded incremental-value experiment.",
        "- A broad cross-position or lifetime-success model remains unsupported by the available targets and cohort coverage.",
    ])

    (output_dir / "phase16_outcome_validation_report.txt").write_text("\n".join(lines), encoding="utf-8")


def write_csv(output_dir: Path, experiment_metrics: dict) -> None:
    rows = []
    for key, result in experiment_metrics.items():
        rows.append(
            {
                "outcome": key,
                "n_eligible": result["n_rows"],
                "baseline_mae": result["baseline"]["mae"],
                "baseline_rmse": result["baseline"]["rmse"],
                "baseline_r2": result["baseline"]["r2"],
                "extended_mae": result["extended"]["mae"],
                "extended_rmse": result["extended"]["rmse"],
                "extended_r2": result["extended"]["r2"],
                "leakage_check": result["leakage_check"],
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "phase16_outcome_validation.csv", index=False)


def run_self_check(validated_df: pd.DataFrame, output_dir: Path) -> None:
    required = [
        output_dir / "phase16_outcome_validation_report.txt",
        output_dir / "phase16_outcome_validation.csv",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required output: {path.name}")
        if path.stat().st_size == 0:
            raise ValueError(f"Required output is empty: {path.name}")
    if validated_df.empty:
        raise ValueError("Validated WR 2025 dataset is empty.")
    if validated_df["season"].nunique() != 1:
        raise ValueError("Validated dataset is not confined to a single season.")
    if int(validated_df["season"].iloc[0]) != 2025:
        raise ValueError("Validated dataset is not the 2025 WR season sample.")
    if validated_df["nfl_id"].duplicated().any():
        raise ValueError("Duplicate athlete IDs remain in the analytical sample.")
    print("SELF_CHECK_OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 16 outcome validation and limited predictive value check.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    games, players, combine, player_play, athlete_table = load_inputs(args.data_dir, args.output_dir)
    validated_df = build_validated_dataset(games, players, combine, player_play, athlete_table)
    experiment_metrics = {}
    for target in ["target_count", "route_count"]:
        result = fit_and_evaluate(validated_df, target)
        if result is not None:
            experiment_metrics[target] = result

    if experiment_metrics:
        write_report(args.output_dir, validated_df, experiment_metrics)
        write_csv(args.output_dir, experiment_metrics)
    else:
        write_report(args.output_dir, validated_df, {})
        pd.DataFrame(columns=["outcome", "n_eligible", "baseline_mae", "baseline_rmse", "baseline_r2", "extended_mae", "extended_rmse", "extended_r2", "leakage_check"]).to_csv(args.output_dir / "phase16_outcome_validation.csv", index=False)

    if args.self_check:
        run_self_check(validated_df, args.output_dir)


if __name__ == "__main__":
    main()
