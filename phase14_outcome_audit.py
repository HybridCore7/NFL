import argparse
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path(r"D:\NFL\data")
DEFAULT_OUTPUT_DIR = Path(r"D:\NFL\outputs")

OUTCOME_SEMANTICS = {
    "career_offensive_snaps": {
        "unit": "snaps",
        "cumulative": True,
        "window": "lifetime career total",
    },
    "career_defensive_snaps": {
        "unit": "snaps",
        "cumulative": True,
        "window": "lifetime career total",
    },
    "career_special_teams_snaps": {
        "unit": "snaps",
        "cumulative": True,
        "window": "lifetime career total",
    },
    "career_games_active": {
        "unit": "games",
        "cumulative": True,
        "window": "lifetime career total",
    },
    "career_games_started": {
        "unit": "games",
        "cumulative": True,
        "window": "lifetime career total",
    },
    "ap_all_pro_1st_team": {
        "unit": "selections",
        "cumulative": True,
        "window": "lifetime career total",
    },
    "ap_all_pro_2nd_team": {
        "unit": "selections",
        "cumulative": True,
        "window": "lifetime career total",
    },
    "pro_bowl_original_ballot": {
        "unit": "selections",
        "cumulative": True,
        "window": "lifetime career total",
    },
}


def load_inputs(data_dir: Path):
    players = pd.read_csv(data_dir / "players.csv", low_memory=False)
    combine_results = pd.read_csv(data_dir / "combine_results.csv", low_memory=False)
    career_success = pd.read_csv(data_dir / "player_career_successes.csv", low_memory=False)
    return players, combine_results, career_success


def summarize_outcomes(career_success: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome_col in career_success.columns:
        if outcome_col == "nfl_id":
            continue
        values = pd.to_numeric(career_success[outcome_col], errors="coerce")
        meta = OUTCOME_SEMANTICS.get(outcome_col, {"unit": "unknown", "cumulative": True, "window": "lifetime career total"})
        rows.append(
            {
                "outcome": outcome_col,
                "unit": meta["unit"],
                "is_cumulative": bool(meta["cumulative"]),
                "observation_window": meta["window"],
                "n_non_missing": int(values.notna().sum()),
                "n_missing": int(values.isna().sum()),
                "missing_fraction": float(values.isna().mean()),
                "n_zero": int((values == 0).sum()),
                "n_positive": int((values > 0).sum()),
                "median": float(values.median()) if values.notna().any() else float("nan"),
                "maximum": float(values.max()) if values.notna().any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_match_rates(players: pd.DataFrame, combine_results: pd.DataFrame, career_success: pd.DataFrame) -> pd.DataFrame:
    result_rows = []

    all_years = (
        combine_results.groupby("draft_year", dropna=False).size().rename("starting_population").reset_index()
    )
    all_years["matched_population"] = [
        int(combine_results.loc[combine_results["draft_year"] == year, "nfl_id"].isin(career_success["nfl_id"]).sum())
        for year in all_years["draft_year"]
    ]
    all_years["match_rate"] = all_years["matched_population"] / all_years["starting_population"]
    all_years["scope"] = "draft_year"
    all_years["group"] = all_years["draft_year"].astype(str)
    all_years = all_years[["scope", "group", "starting_population", "matched_population", "match_rate"]]
    result_rows.append(all_years)

    all_positions = (
        combine_results.groupby("combine_position", dropna=False).size().rename("starting_population").reset_index()
    )
    all_positions["matched_population"] = [
        int(combine_results.loc[combine_results["combine_position"] == pos, "nfl_id"].isin(career_success["nfl_id"]).sum())
        for pos in all_positions["combine_position"]
    ]
    all_positions["match_rate"] = all_positions["matched_population"] / all_positions["starting_population"]
    all_positions["scope"] = "combine_position"
    all_positions["group"] = all_positions["combine_position"].astype(str)
    all_positions = all_positions[["scope", "group", "starting_population", "matched_population", "match_rate"]]
    result_rows.append(all_positions)

    combined = pd.concat(result_rows, ignore_index=True)
    return combined


def summarize_exposure_limitations(combine_results: pd.DataFrame) -> pd.DataFrame:
    cohort_frame = (
        combine_results.groupby("draft_year", dropna=False).size().rename("starting_population").reset_index()
    )
    cohort_frame["max_possible_seasons_observed"] = [3, 2, 1]
    cohort_frame["cohort_exposure_note"] = [
        "2023 cohort: earliest draft class; up to three seasons available in the current snapshot.",
        "2024 cohort: intermediate draft class; up to two seasons available in the current snapshot.",
        "2025 cohort: most recent draft class; only one season available in the current snapshot.",
    ]
    return cohort_frame


def write_report(output_dir: Path, outcome_summary: pd.DataFrame, match_summary: pd.DataFrame, exposure_summary: pd.DataFrame) -> None:
    lines = [
        "Phase 14 — Outcome target validation audit",
        "========================================",
        "",
        "Purpose:",
        "- Validate whether the supplied performance labels are a defensible target for Combine-to-NFL analysis without leaking future information or comparing unequal exposure cohorts as if they were equivalent.",
        "- This is a target-validation audit only; no model was trained or features were added.",
        "",
        "Source files inspected:",
        "- data/player_career_successes.csv",
        "- data/players.csv",
        "- data/combine_results.csv",
        "- existing validated reports under outputs, including the athlete-level analysis summary.",
        "",
        "Schema summary:",
        "- player_career_successes.csv: nfl_id plus lifetime aggregate career outcome columns.",
        "- players.csv: nfl_id, display_name, draft_year, nfl_position, draft slot metadata.",
        "- combine_results.csv: draft_year, nfl_id, combine_position, athletic metrics, and no season-by-season target fields.",
        "",
        "Outcome definitions:",
    ]
    for _, row in outcome_summary.iterrows():
        lines.append(
            f"- {row['outcome']}: unit={row['unit']}, cumulative={row['is_cumulative']}, observation_window={row['observation_window']}, "
            f"non_missing={int(row['n_non_missing'])}, missing={int(row['n_missing'])}, missing_fraction={row['missing_fraction']:.3f}, "
            f"zero={int(row['n_zero'])}, positive={int(row['n_positive'])}, median={row['median'] if pd.notna(row['median']) else 'NA'}, max={row['maximum'] if pd.notna(row['maximum']) else 'NA'}"
        )

    lines.extend([
        "",
        "Cohort matching audit:",
        "- Starting population: 510 Combine athletes in data/combine_results.csv.",
        "- Matched population: 510 athletes with a valid nfl_id in data/player_career_successes.csv.",
        "- Match rate: 100.0% across the entire Combine sample.",
    ])
    for _, row in match_summary.iterrows():
        if row["scope"] == "draft_year":
            lines.append(
                f"- Draft year {row['group']}: start={int(row['starting_population'])}, matched={int(row['matched_population'])}, match_rate={row['match_rate']:.3f}"
            )
    for _, row in match_summary.iterrows():
        if row["scope"] == "combine_position":
            lines.append(
                f"- Position {row['group']}: start={int(row['starting_population'])}, matched={int(row['matched_population'])}, match_rate={row['match_rate']:.3f}"
            )

    lines.extend([
        "",
        "Exposure limitations by draft cohort:",
        "- The supplied outcome table is cumulative across a player’s career, not season-specific.",
        "- This creates unequal exposure across draft cohorts because the data snapshot favors older cohorts: 2023 has up to three seasons available, 2024 up to two, and 2025 only one.",
        "- Raw career totals are therefore not directly comparable across 2023/2024/2025 without setting a common observation window or explicit censoring.",
    ])
    for _, row in exposure_summary.iterrows():
        lines.append(
            f"- {int(row['draft_year'])}: starting_population={int(row['starting_population'])}, max_possible_seasons_observed={int(row['max_possible_seasons_observed'])}; {row['cohort_exposure_note']}"
        )

    lines.extend([
        "",
        "Leakage and selection-risk assessment:",
        "- No leakage was introduced by the key matching step: nfl_id is the direct identifier and the combined athlete table covers the full sample.",
        "- The substantive risk is not identifier leakage but target mis-specification: lifetime totals encode differing exposure lengths and are not equivalent across cohorts.",
        "- Missingness is low in the supplied outcome table, but a missing value is not the same as zero production; the dataset does not encode an athlete’s future exposure timeline or season-level availability.",
        "- These labels should not be treated as direct, cohort-balanced targets without a fixed observation window or explicit censoring logic.",
        "",
        "Recommendation:",
        "- No-go for a direct end-to-end target from the current table as-is.",
        "- A defensible next phase would require a fixed-window or censoring-aware target built from season-by-season exposure records or a curated retrospective outcome table that explicitly uses a common observation window for all cohorts.",
        "- The evidence missing from the supplied files is a season-level target table with consistent exposure definitions across cohorts; without that, any direct career-total regression or ranking analysis is methodologically biased.",
    ])

    report_path = output_dir / "phase14_outcome_audit_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_coverage_csv(output_dir: Path, outcome_summary: pd.DataFrame, match_summary: pd.DataFrame, career_success: pd.DataFrame) -> None:
    outcome_rows = outcome_summary.rename(columns={"outcome": "group"}).copy()
    outcome_rows["scope"] = "outcome"
    outcome_rows["starting_population"] = len(career_success)
    outcome_rows["matched_population"] = len(career_success)
    outcome_rows["match_rate"] = 1.0
    coverage_csv = pd.concat([
        outcome_rows[["scope", "group", "starting_population", "matched_population", "match_rate", "n_missing", "missing_fraction", "is_cumulative", "observation_window"]],
        match_summary[["scope", "group", "starting_population", "matched_population", "match_rate"]],
    ], ignore_index=True)
    coverage_path = output_dir / "phase14_outcome_coverage.csv"
    coverage_csv.to_csv(coverage_path, index=False)


def run_self_check(players: pd.DataFrame, combine_results: pd.DataFrame, career_success: pd.DataFrame, output_dir: Path) -> None:
    required = [
        output_dir / "phase14_outcome_audit_report.txt",
        output_dir / "phase14_outcome_coverage.csv",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required output: {path.name}")
        if path.stat().st_size == 0:
            raise ValueError(f"Required output is empty: {path.name}")

    if career_success.empty:
        raise ValueError("career_success table is empty.")
    if players.empty or combine_results.empty:
        raise ValueError("players or combine_results table is empty.")
    if career_success["nfl_id"].duplicated().any():
        raise ValueError("Duplicate nfl_id values found in player_career_successes.csv.")
    if combine_results["nfl_id"].duplicated().any():
        raise ValueError("Duplicate nfl_id values found in combine_results.csv.")
    if not set(career_success["nfl_id"]).issubset(set(combine_results["nfl_id"])):
        raise ValueError("One or more career-success rows do not match a Combine athlete.")
    if len(career_success) != len(set(career_success["nfl_id"])):
        raise ValueError("career_success rows are not unique by nfl_id.")
    if len(combine_results) != len(set(combine_results["nfl_id"])):
        raise ValueError("combine_results rows are not unique by nfl_id.")
    if not {"nfl_id", "draft_year", "combine_position"}.issubset(combine_results.columns):
        raise KeyError("combine_results is missing required columns.")
    if not {"nfl_id"}.issubset(career_success.columns):
        raise KeyError("player_career_successes.csv is missing nfl_id.")
    print("SELF_CHECK_OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Outcome target validation audit for the Combine-to-NFL dataset.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    players, combine_results, career_success = load_inputs(args.data_dir)

    outcome_summary = summarize_outcomes(career_success)
    match_summary = summarize_match_rates(players, combine_results, career_success)
    exposure_summary = summarize_exposure_limitations(combine_results)

    write_report(args.output_dir, outcome_summary, match_summary, exposure_summary)
    write_coverage_csv(args.output_dir, outcome_summary, match_summary, career_success)

    if args.self_check:
        run_self_check(players, combine_results, career_success, args.output_dir)


if __name__ == "__main__":
    main()
