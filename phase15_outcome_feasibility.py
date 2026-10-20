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
    tracking_samples = {}
    for year in [2023, 2024, 2025]:
        sample = pd.read_csv(
            data_dir / f"game_tracking_{year}.csv",
            usecols=["game_id", "play_id", "nfl_id", "event"],
            low_memory=False,
            nrows=2000,
        )
        tracking_samples[year] = {
            "event_values": sorted(sample["event"].dropna().astype(str).unique().tolist())[:10],
            "non_null_event_rows": int(sample["event"].notna().sum()),
        }
    return games, players, combine, player_play, tracking_samples


def summarize_coverage(games: pd.DataFrame, players: pd.DataFrame, combine: pd.DataFrame, player_play: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    season_table = player_play.merge(games[["game_id", "season"]], on="game_id", how="left")
    season_table = season_table.merge(players[["nfl_id", "nfl_position"]], on="nfl_id", how="left")
    season_table = season_table.merge(combine[["nfl_id", "combine_position"]], on="nfl_id", how="left")
    season_table["season"] = pd.to_numeric(season_table["season"], errors="coerce")
    season_table["target_num"] = pd.to_numeric(season_table["target"], errors="coerce").fillna(0).astype(int)
    season_table["route_flag"] = season_table["route_ran"].notna().astype(int)
    season_table["tackle_num"] = pd.to_numeric(season_table["tackle"], errors="coerce").fillna(0).astype(int)
    season_table["assist_num"] = pd.to_numeric(season_table["assist"], errors="coerce").fillna(0).astype(int)

    coverage_rows = []
    for season in sorted(season_table["season"].dropna().unique().tolist()):
        season_df = season_table[season_table["season"] == season]
        coverage_rows.append({
            "season": int(season),
            "n_player_play_rows": int(len(season_df)),
            "n_unique_players": int(season_df["nfl_id"].nunique()),
            "n_combine_players": int(season_df["nfl_id"].isin(combine["nfl_id"]).sum()),
            "n_unique_combine_players": int(season_df.loc[season_df["nfl_id"].isin(combine["nfl_id"]), "nfl_id"].nunique()),
        })
    coverage_summary = pd.DataFrame(coverage_rows).sort_values("season").reset_index(drop=True)

    candidate_rows = []
    candidate_specs = [
        ("wr_route_count_season", "WR", "routes", "count(route_ran not null)", "player-season", "WR on a given season", "route_ran non-null rows"),
        ("wr_target_count_season", "WR", "targets", "sum(target==1)", "player-season", "WR on a given season", "player_play target flag"),
        ("te_route_count_season", "TE", "routes", "count(route_ran not null)", "player-season", "TE on a given season", "route_ran non-null rows"),
        ("db_tackle_plus_assist_count_season", "DB", "tackles_plus_assists", "sum(tackle + assist)", "player-season", "DB on a given season", "player_play tackle and assist counts"),
    ]
    for candidate_name, position_label, unit, formula, window, scope, note in candidate_specs:
        if position_label == "WR":
            position_df = season_table[season_table["combine_position"] == "WR"].copy()
            if candidate_name.endswith("route_count_season"):
                summary = position_df.groupby(["season", "nfl_id"], dropna=False)["route_flag"].sum().reset_index(name="value")
            else:
                summary = position_df.groupby(["season", "nfl_id"], dropna=False)["target_num"].sum().reset_index(name="value")
        elif position_label == "TE":
            position_df = season_table[season_table["combine_position"] == "TE"].copy()
            summary = position_df.groupby(["season", "nfl_id"], dropna=False)["route_flag"].sum().reset_index(name="value")
        else:
            position_df = season_table[season_table["combine_position"] == "DB"].copy()
            summary = (position_df.groupby(["season", "nfl_id"], dropna=False)[["tackle_num", "assist_num"]].sum().reset_index())
            summary["value"] = summary["tackle_num"] + summary["assist_num"]
            summary = summary[["season", "nfl_id", "value"]]

        distinct = summary.groupby("season")["nfl_id"].nunique().reset_index(name="n_unique_athletes")
        total_players = int(summary["nfl_id"].nunique())
        candidate_rows.append({
            "candidate_name": candidate_name,
            "position_scope": position_label,
            "unit": unit,
            "formula": formula,
            "observation_window": window,
            "n_unique_athlete_seasons": int(len(summary)),
            "n_unique_combine_athletes": total_players,
            "coverage_2023": int(distinct.loc[distinct["season"] == 2023, "n_unique_athletes"].sum()) if 2023 in distinct["season"].tolist() else 0,
            "coverage_2024": int(distinct.loc[distinct["season"] == 2024, "n_unique_athletes"].sum()) if 2024 in distinct["season"].tolist() else 0,
            "coverage_2025": int(distinct.loc[distinct["season"] == 2025, "n_unique_athletes"].sum()) if 2025 in distinct["season"].tolist() else 0,
            "scope": scope,
            "note": note,
        })

    candidate_summary = pd.DataFrame(candidate_rows)
    return coverage_summary, candidate_summary


def write_outputs(output_dir: Path, coverage_summary: pd.DataFrame, candidate_summary: pd.DataFrame, tracking_samples: dict) -> None:
    report_lines = [
        "Phase 15 — Season-level NFL outcome feasibility audit",
        "====================================================",
        "",
        "Objective:",
        "- Determine whether the supplied season-level play data support a defensible NFL outcome that can be linked to a validated Combine movement feature set.",
        "- This is a feasibility audit, not a modeling phase.",
        "",
        "Data inspected:",
        "- data/player_play.csv",
        "- data/games.csv",
        "- data/players.csv",
        "- data/combine_results.csv",
        "- schema samples from game_tracking_2023.csv, game_tracking_2024.csv, and game_tracking_2025.csv.",
        "",
        "Season and identifier coverage:",
        f"- games.csv contains seasons {sorted(coverage_summary['season'].unique().tolist())}.",
        "- player_play.csv contains per-play player entries keyed by game_id + play_id + nfl_id; it is not a player-season table by itself.",
        "- The play-level table already shows the main issue: it records participation and event instances, not a single pre-computed NFL performance target.",
        "",
        "Tracking-event semantics:",
    ]
    for year, sample in tracking_samples.items():
        report_lines.append(
            f"- game_tracking_{year}.csv sample events: {sample['event_values']} (non-null event rows sampled={sample['non_null_event_rows']})"
        )
    report_lines.extend([
        "",
        "Candidate outcome definitions:",
        "- player-season WR route count = count of rows where route_ran is not null for a given nfl_id-season.",
        "- player-season WR target count = sum of target indicators for the same nfl_id-season.",
        "- player-season TE route count = count of rows where route_ran is not null for a given nfl_id-season.",
        "- player-season DB tackle-plus-assist count = sum of tackle + assist for a given nfl_id-season.",
        "",
        "Why these are not universally comparable:",
        "- They are position-specific, not a single cross-position outcome.",
        "- route_ran and target are valid for pass-catchers but not for all Combine athletes.",
        "- tackle and assist are valid defensive event totals but do not map cleanly onto the pass-catching movement features used for WR/DB comparisons.",
        "- A season is a consistent time window only within a single NFL season; it does not solve the cohort exposure problem across 2023, 2024, and 2025 when multi-season totals are used.",
        "",
        "Candidate coverage summary:",
    ])
    for _, row in candidate_summary.iterrows():
        report_lines.append(
            f"- {row['candidate_name']}: position={row['position_scope']}, unit={row['unit']}, window={row['observation_window']}, athlete_seasons={int(row['n_unique_athlete_seasons'])}, athletes={int(row['n_unique_combine_athletes'])}, 2023={int(row['coverage_2023'])}, 2024={int(row['coverage_2024'])}, 2025={int(row['coverage_2025'])}"
        )

    report_lines.extend([
        "",
        "Decision:",
        "- CONDITIONAL GO: season-bounded, position-specific outcomes are feasible for WR/TE route opportunities and for DB tackle-plus-assist counts, but only within a specific role and after explicit validation.",
        "- This is not a universal outcome for all Combine athletes and should not be treated as a general NFL performance target across the full cohort.",
        "- No single defensible target exists today for the full 2023-2025 Combine sample without either a position-specific target pipeline or a curated season-by-season outcome table.",
        "- The current data support a more limited use case: bounded role-specific outcomes for pass-catchers or defenders, not a single cross-position prediction target.",
        "",
        "Risks and caveats:",
        "- player_play rows are not equivalent to snaps or routes for all players; the schema does not encode a universal 'snap count' field.",
        "- route_ran is meaningful for pass routes, but team play design, target share, and role still create selection effects.",
        "- season-level counts can vary by team usage and injury recovery; they are a better bounded outcome than career totals but remain imperfect proxies.",
        "- Any future modeling must keep all target construction and filtering within the season-specific frame and avoid leaking future season information into earlier cohorts.",
    ])

    (output_dir / "phase15_outcome_feasibility_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    candidate_summary.to_csv(output_dir / "phase15_outcome_feasibility.csv", index=False)


def run_self_check(output_dir: Path, coverage_summary: pd.DataFrame, candidate_summary: pd.DataFrame) -> None:
    required_files = [
        output_dir / "phase15_outcome_feasibility_report.txt",
        output_dir / "phase15_outcome_feasibility.csv",
    ]
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing required output: {path.name}")
        if path.stat().st_size == 0:
            raise ValueError(f"Required output is empty: {path.name}")

    if coverage_summary.empty:
        raise ValueError("Season coverage table is empty.")
    if candidate_summary.empty:
        raise ValueError("Candidate outcome table is empty.")
    if set([2023, 2024, 2025]).difference(set(coverage_summary["season"].unique())):
        raise ValueError("Season coverage is missing one or more expected seasons.")
    if candidate_summary["candidate_name"].nunique() < 4:
        raise ValueError("Expected at least four candidate season-level outcome definitions.")
    print("SELF_CHECK_OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 15 season-level outcome feasibility audit.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    games, players, combine, player_play, tracking_samples = load_inputs(args.data_dir)
    coverage_summary, candidate_summary = summarize_coverage(games, players, combine, player_play)
    write_outputs(args.output_dir, coverage_summary, candidate_summary, tracking_samples)

    if args.self_check:
        run_self_check(args.output_dir, coverage_summary, candidate_summary)


if __name__ == "__main__":
    main()
