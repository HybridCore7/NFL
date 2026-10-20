import argparse
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path(r"D:\NFL\data")
DEFAULT_OUTPUT_DIR = Path(r"D:\NFL\outputs")


def load_inputs(data_dir: Path):
    games = pd.read_csv(data_dir / "games.csv", usecols=["game_id", "season"], low_memory=False)
    players = pd.read_csv(data_dir / "players.csv", usecols=["nfl_id", "draft_year", "nfl_position"], low_memory=False)
    combine = pd.read_csv(data_dir / "combine_results.csv", usecols=["nfl_id", "draft_year", "combine_position"], low_memory=False)
    player_play = pd.read_csv(
        data_dir / "player_play.csv",
        usecols=["game_id", "nfl_id", "route_ran", "target", "tackle", "assist", "sack"],
        low_memory=False,
    )
    career_success = pd.read_csv(data_dir / "player_career_successes.csv", low_memory=False)
    athlete_table = pd.read_csv(data_dir.parent / "outputs" / "athlete_level_analysis.csv", low_memory=False)
    return games, players, combine, player_play, career_success, athlete_table


def summarize_career_outcomes(career_success: pd.DataFrame):
    rows = []
    for col in [c for c in career_success.columns if c != "nfl_id"]:
        series = pd.to_numeric(career_success[col], errors="coerce")
        rows.append(
            {
                "outcome_name": col,
                "source_table": "player_career_successes.csv",
                "source_columns": [col],
                "unit": {
                    "career_offensive_snaps": "snaps",
                    "career_defensive_snaps": "snaps",
                    "career_special_teams_snaps": "snaps",
                    "career_games_active": "games",
                    "career_games_started": "games",
                    "ap_all_pro_1st_team": "selections",
                    "ap_all_pro_2nd_team": "selections",
                    "pro_bowl_original_ballot": "selections",
                }.get(col, "unknown"),
                "meaning": {
                    "career_offensive_snaps": "lifetime offensive snap count",
                    "career_defensive_snaps": "lifetime defensive snap count",
                    "career_special_teams_snaps": "lifetime special-teams snap count",
                    "career_games_active": "lifetime active games",
                    "career_games_started": "lifetime games started",
                    "ap_all_pro_1st_team": "career Pro Bowl / All-Pro selection count",
                    "ap_all_pro_2nd_team": "career Pro Bowl / All-Pro selection count",
                    "pro_bowl_original_ballot": "career Pro Bowl ballot count",
                }.get(col, "career aggregate outcome"),
                "aggregation_level": "athlete-lifetime",
                "observation_window": "career total",
                "n_unique_athletes": int(career_success["nfl_id"].nunique()),
                "n_non_missing": int(series.notna().sum()),
                "coverage_pct": float(series.notna().mean() * 100.0),
                "is_cumulative": True,
                "exposure_adjusted": False,
                "category": "success" if col.startswith("ap_") or col.startswith("pro_bowl") else "participation",
            }
        )
    return pd.DataFrame(rows)


def summarize_wr_season_outcomes(games: pd.DataFrame, players: pd.DataFrame, combine: pd.DataFrame, player_play: pd.DataFrame):
    season_play = player_play.merge(games[["game_id", "season"]], on="game_id", how="left")
    season_play = season_play.merge(players[["nfl_id", "draft_year"]], on="nfl_id", how="left")
    season_play = season_play.merge(combine[["nfl_id", "draft_year", "combine_position"]], on=["nfl_id", "draft_year"], how="left")
    season_play["season"] = pd.to_numeric(season_play["season"], errors="coerce")
    season_play["route_flag"] = season_play["route_ran"].notna().astype(int)
    season_play["target_num"] = pd.to_numeric(season_play["target"], errors="coerce").fillna(0).astype(float)

    wr = season_play[season_play["combine_position"] == "WR"].copy()
    route_count = wr.groupby(["season", "nfl_id"], dropna=False)["route_flag"].sum().reset_index(name="route_count_wr")
    target_count = wr.groupby(["season", "nfl_id"], dropna=False)["target_num"].sum().reset_index(name="target_count_wr")

    route_summary = pd.DataFrame([
        {
            "outcome_name": "route_count_wr",
            "source_table": "player_play.csv",
            "source_columns": "route_ran",
            "unit": "plays with route_ran populated",
            "meaning": "season-level route volume proxy",
            "aggregation_level": "athlete-season",
            "observation_window": "season",
            "n_unique_athletes": int(route_count["nfl_id"].nunique()),
            "n_non_missing": int(route_count["route_count_wr"].notna().sum()),
            "coverage_pct": float(route_count["route_count_wr"].notna().mean() * 100.0),
            "is_cumulative": False,
            "exposure_adjusted": False,
            "category": "opportunity",
        }
    ])

    target_summary = pd.DataFrame([
        {
            "outcome_name": "target_count_wr",
            "source_table": "player_play.csv",
            "source_columns": "target",
            "unit": "target count",
            "meaning": "season-level target opportunity proxy",
            "aggregation_level": "athlete-season",
            "observation_window": "season",
            "n_unique_athletes": int(target_count["nfl_id"].nunique()),
            "n_non_missing": int(target_count["target_count_wr"].notna().sum()),
            "coverage_pct": float(target_count["target_count_wr"].notna().mean() * 100.0),
            "is_cumulative": False,
            "exposure_adjusted": False,
            "category": "opportunity",
        }
    ])

    return pd.concat([route_summary, target_summary], ignore_index=True)


def build_catalog(dataset_rows: pd.DataFrame, season_rows: pd.DataFrame, career_rows: pd.DataFrame) -> pd.DataFrame:
    catalog = pd.concat([career_rows, season_rows], ignore_index=True)
    catalog = catalog.drop_duplicates(subset=["outcome_name"]).copy()
    return catalog


def compute_option_summary(games, players, combine, player_play, career_success):
    season_play = player_play.merge(games[["game_id", "season"]], on="game_id", how="left")
    season_play = season_play.merge(players[["nfl_id", "draft_year"]], on="nfl_id", how="left")
    season_play = season_play.merge(combine[["nfl_id", "draft_year", "combine_position"]], on=["nfl_id", "draft_year"], how="left")
    season_play["season"] = pd.to_numeric(season_play["season"], errors="coerce")

    wr_2025 = season_play[(season_play["combine_position"] == "WR") & (season_play["season"] == 2025)].copy()
    wr_2025_counts = wr_2025.groupby("nfl_id").size().reset_index(name="play_rows")
    wr_2025_target = wr_2025.groupby("nfl_id")["target"].apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()).reset_index(name="target_total")
    wr_2025_routes = wr_2025.groupby("nfl_id")["route_ran"].apply(lambda s: s.notna().sum()).reset_index(name="routes_total")

    option_rows = [
        {
            "option": "A. Descriptive Combine movement profile by position",
            "sample_size": int((pd.read_csv(DEFAULT_DATA_DIR.parent / "outputs" / "athlete_level_analysis.csv", low_memory=False)["combine_position"].notna().sum())),
            "feasibility": "Feasible",
            "validity_risks": "Descriptive only; no outcome target, no causal claim, no predictive translation.",
            "competition_relevance": "High for athlete profiling with clear caveats.",
        },
        {
            "option": "B. Season-bounded Combine-to-NFL association using a validated role-specific outcome",
            "sample_size": int(wr_2025["nfl_id"].nunique()),
            "feasibility": "Conditionally feasible if target semantics are explicitly validated and exposure is bounded.",
            "validity_risks": "Target is an opportunity proxy, not a true route count; role-specific; exposure still limited.",
            "competition_relevance": "Moderate, still bounded and not generalizable.",
        },
        {
            "option": "C. Predictive model comparing conventional metrics and movement features",
            "sample_size": 78,
            "feasibility": "Not currently justified",
            "validity_risks": "Target proxy, tiny sample, unstable train/test split, no defensible improvement signal, no external validation.",
            "competition_relevance": "Low; weak evidence and high model risk.",
        },
    ]
    return pd.DataFrame(option_rows)


def _sources_string(value):
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, tuple):
        return ", ".join(str(v) for v in value)
    if value is None:
        return ""
    return str(value)


def write_report(output_dir: Path, catalog: pd.DataFrame, options: pd.DataFrame):
    lines = [
        "Phase 19 — Research Direction and Outcome Design Gate",
        "====================================================",
        "",
        "Project history summary:",
        "- The project began by auditing the data schema, join keys, and data quality risks before trusting any analysis.",
        "- It validated the Combine movement extraction pipeline and then built athlete-level movement summaries and descriptive position-level profiles.",
        "- It explicitly rejected unsupported route-transfer claims between Combine movement and NFL route semantics.",
        "- It audited outcome targets and rejected raw lifetime totals as non-comparable global targets.",
        "- It tested a bounded season-specific WR outcome approach and found the sample insufficient for a defensible model comparison.",
        "- It closed with an evidence-based stop: descriptive analysis is valid, while predictive modeling remains intentionally deferred.",
        "",
        "Research question reconstruction:",
        "- Original competition objective: assess whether Combine athletic measurements and movement-derived features contain information relevant to future NFL performance.",
        "- Current hypothesis: a defensible subset of Combine movement features may add signal beyond standard Combine measures for a narrow 2025 WR opportunity target.",
        "- Supported by evidence: movement features are fully available for the 2025 WR sample, and the athlete-level movement pipeline is validated and descriptive.",
        "- Unsupported: broad claims that Combine movement has general predictive value for NFL success; any model-level causal claim; any route-transfer equivalence claim between Combine movement and NFL route semantics.",
        "- Disproven: a broad lifetime career-total target and a full conventional baseline that includes three_cone, short_shuttle, and bench_reps are not defensible in this dataset.",
        "",
        "Outcome audit (source-data validation):",
        "- player_play.csv is a play-level table keyed by game_id, nfl_id, and play_id; it is not a pre-aggregated athlete-season table.",
        "- target is a binary indicator of whether a row records a target, not a validated receptions or route count field in the raw schema.",
        "- route_ran is a route label field, not an independent measure of actual snaps, receiving volume, or route opportunity across all players and positions.",
        "- player_career_successes.csv holds lifetime cumulative career totals and is therefore not exposure-normalized across draft cohorts.",
        "- players.csv and games.csv provide athlete identity, draft year, and season metadata, but no season-specific complete outcome table for the full Combine sample.",
        "",
        "Candidate outcomes:",
    ]
    for _, row in catalog.iterrows():
        lines.append(
            f"- {row['outcome_name']}: source={row['source_table']} / {_sources_string(row['source_columns'])}; unit={row['unit']}; level={row['aggregation_level']}; window={row['observation_window']}; coverage={row['n_non_missing']}/{row['n_unique_athletes']} ({row['coverage_pct']:.1f}%); category={row['category']}; exposure_adjusted={row['exposure_adjusted']}"
        )

    lines.extend([
        "",
        "Research options:",
    ])
    for _, row in options.iterrows():
        lines.append(
            f"- {row['option']}: sample={row['sample_size']}; feasibility={row['feasibility']}; risks={row['validity_risks']}; relevance={row['competition_relevance']}"
        )

    lines.extend([
        "",
        "Decision criteria for a predictive phase:",
        "- target must be validated as a clear football outcome, not merely a proxy",
        "- usable sample must be large enough for stable train/test evaluation",
        "- missingness must remain manageable without arbitrary imputation at the full-cohort level",
        "- exposure and season comparability must be explicit and bounded",
        "- train/test design must avoid leakage and preserve the same athlete set across model comparisons",
        "- result must be interpretable as a bounded analysis, not a broad scouting claim",
        "",
        "Recommendation:",
        "- The strongest defensible direction is descriptive analysis of validated Combine movement profiles by position, using the athlete-level movement features that have already been checked for measurement validity.",
        "- The current predictive model question is not ready for a new modeling phase because the target remains a bounded opportunity proxy, the sample is modest, the split is small, and the repeated split evidence is unstable.",
        "- Do not force a model. Narrow descriptive analysis and a clearer outcome design are the evidence-supported next steps.",
        "",
        "GO: DESCRIPTIVE ANALYSIS",
    ])
    (output_dir / "phase19_research_direction_report.txt").write_text("\n".join(lines), encoding="utf-8")


def write_outcome_catalog(output_dir: Path, catalog: pd.DataFrame):
    catalog.to_csv(output_dir / "phase19_outcome_catalog.csv", index=False)


def run_self_check(catalog: pd.DataFrame, output_dir: Path):
    required = [
        output_dir / "phase19_research_direction_report.txt",
        output_dir / "phase19_outcome_catalog.csv",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required output: {path.name}")
        if path.stat().st_size == 0:
            raise ValueError(f"Required output is empty: {path.name}")
    if catalog.empty:
        raise ValueError("Outcome catalog is empty.")
    if catalog["outcome_name"].isna().any():
        raise ValueError("Outcome catalog contains missing names.")
    if catalog["outcome_name"].duplicated().any():
        raise ValueError("Outcome catalog contains duplicate names.")
    print("SELF_CHECK_OK")
    print("catalog_rows", len(catalog))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 19 research direction and outcome design gate.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    games, players, combine, player_play, career_success, athlete_table = load_inputs(args.data_dir)
    career_rows = summarize_career_outcomes(career_success)
    season_rows = summarize_wr_season_outcomes(games, players, combine, player_play)
    catalog = build_catalog([], season_rows, career_rows)
    options = compute_option_summary(games, players, combine, player_play, career_success)
    write_report(args.output_dir, catalog, options)
    write_outcome_catalog(args.output_dir, catalog)

    if args.self_check:
        run_self_check(catalog, args.output_dir)


if __name__ == "__main__":
    main()
