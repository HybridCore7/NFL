import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

DATA_DIR = Path(r"D:\NFL\data")
OUTPUT_DIR = Path(r"D:\NFL\outputs")


def pearson_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~(pd.isna(x) | pd.isna(y))
    x = x[valid]
    y = y[valid]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, 0
    r, _ = pearsonr(x, y)
    z = 0.5 * math.log((1.0 + r) / (1.0 - r)) if abs(r) < 1.0 else np.sign(r) * 100.0
    se = 1.0 / math.sqrt(len(x) - 3)
    low = math.tanh(z - 1.96 * se)
    high = math.tanh(z + 1.96 * se)
    return float(r), float(low), float(high), int(len(x))


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, n_bootstrap: int = 2000, seed: int = 42):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~(pd.isna(x) | pd.isna(y))
    x = x[valid]
    y = y[valid]
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


def load_wr_hitch_plays(data_dir: Path) -> pd.DataFrame:
    player_play = pd.read_csv(data_dir / "player_play.csv", usecols=["game_id", "play_id", "nfl_id", "route_ran"], low_memory=False)
    players = pd.read_csv(data_dir / "players.csv", usecols=["nfl_id", "nfl_position"], low_memory=False)
    player_play["nfl_id"] = pd.to_numeric(player_play["nfl_id"], errors="coerce")
    players["nfl_id"] = pd.to_numeric(players["nfl_id"], errors="coerce")
    route_df = player_play.merge(players, on="nfl_id", how="left")
    hitch = route_df[(route_df["nfl_position"] == "WR") & (route_df["route_ran"] == "HITCH")][["game_id", "play_id", "nfl_id"]].drop_duplicates()
    return hitch.reset_index(drop=True)


def load_combine_comeback_feature(output_dir: Path) -> pd.DataFrame:
    comb = pd.read_csv(output_dir / "combine_movement_features.csv", low_memory=False)
    comb["nfl_id"] = pd.to_numeric(comb["nfl_id"], errors="coerce")
    subset = comb[
        (comb["combine_position"] == "WR")
        & (comb["drill_name"] == "COMEBACK_ROUTE_RIGHT")
        & (comb["feature_name"] == "early_accel_1s")
        & (comb["feature_value"].notna())
    ].copy()
    subset = subset[["nfl_id", "feature_value"]].drop_duplicates()
    return subset.groupby("nfl_id", as_index=False)["feature_value"].median().rename(columns={"feature_value": "combine_early_accel_1s"})


def route_transfer_readiness(data_dir: Path, output_dir: Path) -> dict:
    hitch = load_wr_hitch_plays(data_dir)
    combine = load_combine_comeback_feature(output_dir)
    overlap = set(hitch["nfl_id"]).intersection(set(combine["nfl_id"]))

    yearly = []
    for year in [2023, 2024, 2025]:
        tracking = pd.read_csv(data_dir / f"game_tracking_{year}.csv", usecols=["game_id", "play_id", "nfl_id", "event"], low_memory=False)
        tracking["game_id"] = pd.to_numeric(tracking["game_id"], errors="coerce")
        tracking["play_id"] = pd.to_numeric(tracking["play_id"], errors="coerce")
        tracking["nfl_id"] = pd.to_numeric(tracking["nfl_id"], errors="coerce")
        tracking["event"] = tracking["event"].fillna("").astype(str)
        rel = tracking.merge(hitch, on=["game_id", "play_id", "nfl_id"], how="inner")
        yearly.append({
            "year": year,
            "unique_hitch_plays": int(rel[["game_id", "play_id", "nfl_id"]].drop_duplicates().shape[0]),
            "ball_snap_events": int(rel["event"].eq("ball_snap").sum()),
            "pass_forward_events": int(rel["event"].eq("pass_forward").sum()),
        })

    return {
        "wr_hitch_route_plays": int(len(hitch)),
        "wr_hitch_unique_players": int(hitch["nfl_id"].nunique()),
        "combine_comeback_unique_players": int(combine["nfl_id"].nunique()),
        "matched_players": int(len(overlap)),
        "yearly_summary": yearly,
        "blocker": "The route label exists, but the NFL tracking data do not define an explicit route-break event equivalent to the Combine COMEBACK_ROUTE_RIGHT drill, so no defensible movement-transfer metric should be claimed.",
    }


def compute_route_early_accel(play_df: pd.DataFrame, window_seconds: float) -> float:
    play_df = play_df.sort_values("time_dt").copy()
    if len(play_df) < 3:
        return np.nan
    if "ball_snap" not in play_df["event"].fillna("").astype(str).values:
        return np.nan
    snap_row = play_df[play_df["event"] == "ball_snap"].head(1)
    if snap_row.empty:
        return np.nan
    snap_time = pd.to_datetime(snap_row["time_dt"].iloc[0])
    window = play_df[(play_df["time_dt"] >= snap_time) & (play_df["time_dt"] <= snap_time + pd.Timedelta(seconds=window_seconds))].copy()
    if len(window) < 3:
        return np.nan
    window = window.dropna(subset=["x_num", "y_num", "s_num", "time_dt"]).copy()
    if len(window) < 3:
        return np.nan
    time_vals = (window["time_dt"] - snap_time).dt.total_seconds().to_numpy(float)
    if np.any(np.diff(time_vals) <= 0):
        return np.nan
    s = pd.to_numeric(window["s_num"], errors="coerce").to_numpy(float)
    if not np.isfinite(s).all() or not np.isfinite(time_vals).all():
        return np.nan
    elapsed = float(time_vals[-1] - time_vals[0])
    if elapsed <= 0:
        return np.nan
    early_speed = float(np.nanmean(s))
    start_speed = float(s[0])
    return float((early_speed - start_speed) / elapsed)


def quantify_route_transfer(data_dir: Path, output_dir: Path, windows=(0.75, 1.0, 1.25, 1.5)) -> pd.DataFrame:
    hitch = load_wr_hitch_plays(data_dir)
    combine = load_combine_comeback_feature(output_dir)
    valid_rows = []

    for year in [2023, 2024, 2025]:
        tracking = pd.read_csv(data_dir / f"game_tracking_{year}.csv", usecols=["game_id", "play_id", "nfl_id", "time", "x", "y", "s", "event"], low_memory=False)
        tracking["game_id"] = pd.to_numeric(tracking["game_id"], errors="coerce")
        tracking["play_id"] = pd.to_numeric(tracking["play_id"], errors="coerce")
        tracking["nfl_id"] = pd.to_numeric(tracking["nfl_id"], errors="coerce")
        tracking = tracking[tracking["game_id"].astype(str).str.isnumeric()].copy()
        tracking["time_dt"] = pd.to_datetime(tracking["time"], errors="coerce")
        tracking["x_num"] = pd.to_numeric(tracking["x"], errors="coerce")
        tracking["y_num"] = pd.to_numeric(tracking["y"], errors="coerce")
        tracking["s_num"] = pd.to_numeric(tracking["s"], errors="coerce")
        tracking["event"] = tracking["event"].fillna("").astype(str)
        hitch["game_id"] = pd.to_numeric(hitch["game_id"], errors="coerce")
        hitch["play_id"] = pd.to_numeric(hitch["play_id"], errors="coerce")
        hitch["nfl_id"] = pd.to_numeric(hitch["nfl_id"], errors="coerce")
        relevant = tracking.merge(hitch, on=["game_id", "play_id", "nfl_id"], how="inner")
        if relevant.empty:
            continue

        for window_seconds in windows:
            per_player = []
            for key, g in relevant.groupby(["nfl_id"]):
                vals = []
                for _, play in g.groupby(["game_id", "play_id"]):
                    val = compute_route_early_accel(play, window_seconds)
                    if np.isfinite(val):
                        vals.append(val)
                if len(vals):
                    per_player.append({"nfl_id": key, "window_seconds": float(window_seconds), "route_early_accel_median": float(np.nanmedian(vals)), "n_valid_hitch_plays": int(len(vals))})
            if len(per_player):
                route_df = pd.DataFrame(per_player)
                route_df["nfl_id"] = pd.to_numeric(route_df["nfl_id"], errors="coerce")
                merged = route_df.merge(combine, on="nfl_id", how="inner")
                if len(merged) >= 3:
                    pearson_r, low, high, n = pearson_ci(merged["route_early_accel_median"].to_numpy(), merged["combine_early_accel_1s"].to_numpy())
                    rho, rho_low, rho_high = bootstrap_spearman(merged["route_early_accel_median"].to_numpy(), merged["combine_early_accel_1s"].to_numpy(), n_bootstrap=1000, seed=42)
                    valid_rows.append({
                        "window_seconds": window_seconds,
                        "matched_players": int(len(merged)),
                        "pearson_r": pearson_r,
                        "pearson_ci_low": low,
                        "pearson_ci_high": high,
                        "spearman_rho": rho,
                        "spearman_ci_low": rho_low,
                        "spearman_ci_high": rho_high,
                        "n_valid_hitch_plays_total": int(route_df["n_valid_hitch_plays"].sum()),
                        "route_feature_median": float(route_df["route_early_accel_median"].median()),
                        "combine_feature_median": float(merged["combine_early_accel_1s"].median()),
                    })

    if not valid_rows:
        return pd.DataFrame(
            columns=[
                "window_seconds",
                "matched_players",
                "pearson_r",
                "pearson_ci_low",
                "pearson_ci_high",
                "spearman_rho",
                "spearman_ci_low",
                "spearman_ci_high",
                "n_valid_hitch_plays_total",
                "route_feature_median",
                "combine_feature_median",
            ]
        )

    return pd.DataFrame(valid_rows).sort_values("window_seconds").reset_index(drop=True)


def self_check():
    route_df = load_wr_hitch_plays(DATA_DIR)
    combine_df = load_combine_comeback_feature(OUTPUT_DIR)
    overlap = set(route_df["nfl_id"]).intersection(set(combine_df["nfl_id"]))

    print("SELF_CHECK_OK")
    print("WR_HITCH_route_plays", len(route_df))
    print("WR_HITCH_unique_players", route_df["nfl_id"].nunique())
    print("COMEBACK_ROUTE_RIGHT_unique_players", combine_df["nfl_id"].nunique())
    print("overlap_players", len(overlap))

    readiness = route_transfer_readiness(DATA_DIR, OUTPUT_DIR)
    print("matched_players", readiness["matched_players"])
    print("blocker", readiness["blocker"])
    for year_summary in readiness["yearly_summary"]:
        print(year_summary)


def main():
    parser = argparse.ArgumentParser(description="Phase 12: Combine-to-NFL WR route transfer audit")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return

    readiness = route_transfer_readiness(args.data_dir, args.output_dir)
    print(readiness)


if __name__ == "__main__":
    main()
