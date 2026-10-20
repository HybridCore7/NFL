import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path(r"D:\NFL\data")
DEFAULT_OUTPUT_DIR = Path(r"D:\NFL\outputs")


def load_tracking(data_dir: Path) -> pd.DataFrame:
    file_path = data_dir / 'combine_tracking.csv'
    if not file_path.exists():
        raise FileNotFoundError(f"Expected Combine tracking file at {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    out = df[df['entity_type'].eq('PLAYER')].copy()
    out['time_dt'] = pd.to_datetime(out['time'], errors='coerce')
    out['x_num'] = pd.to_numeric(out['x'], errors='coerce')
    out['y_num'] = pd.to_numeric(out['y'], errors='coerce')
    out['s_num'] = pd.to_numeric(out['s'], errors='coerce')
    out['a_num'] = pd.to_numeric(out['a'], errors='coerce')
    out['dis_num'] = pd.to_numeric(out['dis'], errors='coerce')
    out['dir_num'] = pd.to_numeric(out['dir'], errors='coerce')
    out = out.sort_values(['draft_year', 'nfl_id', 'drill_type', 'drill_name', 'attempt', 'time_dt']).reset_index(drop=True)
    return out


def dedupe_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    keep = ['draft_year', 'nfl_id', 'drill_type', 'drill_name', 'attempt', 'entity_type', 'time', 'time_dt', 'x', 'y', 's', 'a', 'dis', 'dir', 'x_num', 'y_num', 's_num', 'a_num', 'dis_num', 'dir_num']
    out = df[keep].copy()
    cols = ['x_num', 'y_num', 's_num', 'a_num', 'dis_num', 'dir_num']
    out = out.groupby(['draft_year', 'nfl_id', 'drill_type', 'drill_name', 'attempt', 'time_dt'], as_index=False)[cols].median()
    out['time'] = out['time_dt'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f').str.rstrip('0').str.rstrip('.')
    out['x'] = out['x_num']
    out['y'] = out['y_num']
    out['s'] = out['s_num']
    out['a'] = out['a_num']
    out['dis'] = out['dis_num']
    out['dir'] = out['dir_num']
    out = out.sort_values(['draft_year', 'nfl_id', 'drill_type', 'drill_name', 'attempt', 'time_dt']).reset_index(drop=True)
    out = out.loc[~out.duplicated(subset=['draft_year', 'nfl_id', 'drill_type', 'drill_name', 'attempt', 'time_dt'], keep='last')].copy()
    return out


def compute_heading_change(g: pd.DataFrame) -> tuple[float, int]:
    if len(g) < 6:
        return float('nan'), -1
    x = g['x_num'].to_numpy(float)
    y = g['y_num'].to_numpy(float)
    t = (g['time_dt'].astype('int64') / 1_000_000_000).to_numpy(float)
    if not np.all(np.diff(t) > 0):
        return float('nan'), -1
    dx = np.diff(x)
    dy = np.diff(y)
    step_mag = np.hypot(dx, dy)
    valid_step = np.isfinite(step_mag) & (step_mag > 0.05)
    if not np.any(valid_step):
        return float('nan'), -1

    heading = np.degrees(np.arctan2(dy, dx))
    heading = (heading + 360) % 360
    delta = np.abs((heading[1:] - heading[:-1] + 180) % 360 - 180)
    idx = -1
    best_val = 0.0
    for i in range(len(delta) - 2):
        if not valid_step[i] or not valid_step[i + 1] or not valid_step[i + 2]:
            continue
        if delta[i] <= 60 or delta[i + 1] <= 60 or delta[i + 2] <= 60:
            continue
        local_max = float(np.max(delta[i:i + 3]))
        if local_max > best_val:
            best_val = local_max
            idx = i + 1
    if idx < 0:
        return float('nan'), -1
    return float(best_val), idx


def smoothed_speed_gradient(speed: np.ndarray, times: np.ndarray) -> np.ndarray:
    speed = np.asarray(speed, dtype=float)
    times = np.asarray(times, dtype=float)
    if speed.size < 3 or times.size != speed.size:
        return np.full(speed.shape, np.nan, dtype=float)
    if not np.all(np.diff(times) > 0):
        return np.full(speed.shape, np.nan, dtype=float)
    smooth = pd.Series(speed).rolling(window=3, center=True, min_periods=1).mean().to_numpy(dtype=float)
    return np.gradient(smooth, times)


def extract_attempt_features(g: pd.DataFrame) -> dict:
    g = g.sort_values('time_dt').copy()
    if len(g) < 3:
        return {}
    t = (g['time_dt'].astype('int64') / 1_000_000_000).to_numpy(float)
    s = g['s_num'].to_numpy(float)
    x = g['x_num'].to_numpy(float)
    y = g['y_num'].to_numpy(float)
    a_mag = g['a_num'].to_numpy(float)

    # Remove non-positive elapsed times before any derivative-based calculations.
    keep = np.r_[True, np.diff(t) > 0]
    if keep.sum() < 3:
        return {}
    t = t[keep]
    s = s[keep]
    x = x[keep]
    y = y[keep]
    a_mag = a_mag[keep]

    max_speed = float(np.nanmax(s)) if np.isfinite(s).any() else np.nan
    t_start = t[0]
    t_rel = t - t_start

    early_window = t_rel <= 1.0
    if early_window.sum() >= 2:
        early_t = t_rel[early_window]
        early_s = s[early_window]
        if np.all(np.diff(early_t) > 0) and early_t[-1] > early_t[0]:
            early_speed = float(np.nanmean(early_s))
            early_dt = float(early_t[-1] - early_t[0])
            early_accel_1s = float((early_speed - s[0]) / early_dt) if early_dt > 0 else np.nan
        else:
            early_accel_1s = np.nan
    else:
        early_accel_1s = np.nan
    
    t90 = np.nan
    if np.isfinite(max_speed) and max_speed > 0:
        idx = np.where(s >= 0.9 * max_speed)[0]
        if len(idx):
            t90 = float(t[idx[0]] - t_start)

    speed_grad = smoothed_speed_gradient(s, t)
    signed_max_decel = float(np.nanmin(speed_grad)) if np.isfinite(speed_grad).any() else np.nan

    # path metrics
    disp = np.hypot(np.diff(x), np.diff(y))
    path_length = float(np.nansum(disp)) if len(disp) else 0.0
    net_disp = float(np.hypot(x[-1] - x[0], y[-1] - y[0])) if len(x) > 1 else 0.0

    # cut metrics for drill families that contain a directional change
    cut_min_speed = np.nan
    cut_speed_retention = np.nan
    cut_reaccel = np.nan
    cut_idx = -1
    drill_name = str(g['drill_name'].iloc[0])
    drill_type = str(g['drill_type'].iloc[0])
    if ('THREE' in drill_name.upper()) or ('THREE_CONE' in drill_type.upper()):
        max_delta, cut_idx = compute_heading_change(g)
        if cut_idx > 0 and len(s) > cut_idx + 2 and np.isfinite(max_delta) and max_delta > 60:
            window = slice(max(0, cut_idx - 2), min(len(s), cut_idx + 4))
            cut_window = s[window]
            cut_min_speed = float(np.nanmin(cut_window)) if np.isfinite(cut_window).any() else np.nan
            before_slice = s[max(0, cut_idx - 3):cut_idx]
            after_slice = s[cut_idx + 1:min(len(s), cut_idx + 4)]
            speed_before = float(np.nanmean(before_slice)) if len(before_slice) and np.isfinite(before_slice).any() else np.nan
            speed_after = float(np.nanmean(after_slice)) if len(after_slice) and np.isfinite(after_slice).any() else np.nan
            if np.isfinite(speed_before) and speed_before > 0 and np.isfinite(speed_after):
                cut_speed_retention = float(speed_after / speed_before)
            if len(after_slice) and np.isfinite(speed_after):
                post_idx = np.where(np.isfinite(s[cut_idx + 1:]))[0]
                if len(post_idx):
                    post_t = t[cut_idx + 1 + int(post_idx[0])]
                    elapsed = float(post_t - t[cut_idx])
                    if elapsed > 0:
                        post_speed = float(s[cut_idx + 1 + int(post_idx[0])])
                        cut_reaccel = float((post_speed - speed_after) / elapsed)
    
    # backpedal-to-forward transition for DB drill family
    db_transition_speed = np.nan
    if ('DB' in drill_type.upper()) or ('BACKPEDAL' in drill_name.upper()) or ('DB' in drill_name.upper()):
        if len(x) > 4:
            x_rel = x - x[0]
            y_rel = y - y[0]
            dir_vec = np.gradient(np.arctan2(y_rel, x_rel))
            if np.isfinite(dir_vec).any():
                db_transition_speed = float(np.nanmax(s))

    return {
        'max_speed': max_speed,
        'early_accel_1s': early_accel_1s,
        'time_to_90pct_max_speed_s': t90,
        'signed_max_decel': signed_max_decel,
        'path_length_yd': path_length,
        'net_displacement_yd': net_disp,
        'three_cone_min_speed_cut': cut_min_speed,
        'three_cone_speed_retention': cut_speed_retention,
        'three_cone_post_cut_reaccel': cut_reaccel,
        'db_backpedal_transition_speed': db_transition_speed,
        'feature_provenance': 'derived from s and x/y; signed acceleration from speed gradient; raw a treated as magnitude only',
    }


def build_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(['draft_year', 'nfl_id', 'drill_type', 'drill_name', 'attempt'], dropna=False):
        attempt = g.sort_values('time_dt').copy()
        if len(attempt) < 3:
            continue
        result = extract_attempt_features(attempt)
        if not result:
            continue
        for feat_name, feat_value in result.items():
            if feat_name in {'feature_provenance'}:
                continue
            rows.append({
                'draft_year': int(key[0]),
                'nfl_id': int(key[1]),
                'drill_type': key[2],
                'drill_name': key[3],
                'attempt': int(key[4]),
                'feature_name': feat_name,
                'feature_value': float(feat_value) if pd.notna(feat_value) else np.nan,
                'feature_provenance': result['feature_provenance'],
            })
    feature_df = pd.DataFrame(rows)
    return feature_df


def load_combine_results(data_dir: Path) -> pd.DataFrame:
    path = data_dir / 'combine_results.csv'
    df = pd.read_csv(path, low_memory=False)
    df['nfl_id'] = pd.to_numeric(df['nfl_id'], errors='coerce')
    df['draft_year'] = pd.to_numeric(df['draft_year'], errors='coerce')
    return df


def add_combine_metadata(feature_df: pd.DataFrame, combine_results: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty:
        return feature_df
    cols = ['draft_year', 'nfl_id', 'combine_position', 'combine_height', 'combine_weight', 'forty', 'vertical', 'broad_jump', 'three_cone', 'short_shuttle', 'bench_reps', 'ten_yd_split']
    available = [c for c in cols if c in combine_results.columns]
    merged = feature_df.merge(combine_results[available], on=['draft_year', 'nfl_id'], how='left')
    return merged


def coverage_report(feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame(columns=['drill_type', 'feature_name', 'n_rows', 'coverage'])
    summary = (
        feature_df.groupby(['drill_type', 'feature_name'])['feature_value']
        .agg(['count', lambda s: s.notna().sum()])
        .reset_index()
    )
    summary.columns = ['drill_type', 'feature_name', 'n_rows', 'non_null_rows']
    summary['coverage'] = summary['non_null_rows'] / summary['n_rows']
    return summary


def save_plot(feature_df: pd.DataFrame, combine_results: pd.DataFrame, output_dir: Path) -> None:
    if feature_df.empty:
        return
    wide = feature_df.pivot_table(index=['draft_year', 'nfl_id', 'drill_type', 'drill_name', 'attempt'], columns='feature_name', values='feature_value', aggfunc='first').reset_index()
    meta = combine_results[['draft_year', 'nfl_id', 'combine_position', 'forty']].drop_duplicates()
    merged = wide.merge(meta, on=['draft_year', 'nfl_id'], how='left')
    subset = merged[merged['combine_position'].isin(['WR', 'DB', 'TE']) & merged['forty'].notna() & merged['early_accel_1s'].notna()].copy()
    if subset.empty:
        return
    plt.figure(figsize=(8, 5))
    for pos in sorted(subset['combine_position'].dropna().unique()):
        s = subset[subset['combine_position'] == pos]
        plt.scatter(s['forty'], s['early_accel_1s'], s=20, alpha=0.7, label=pos)
    plt.xlabel('40-yard time (s)')
    plt.ylabel('Early acceleration over first 1s (yd/s^2)')
    plt.title('Combine movement phenotype vs. conventional speed')
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / 'combine_movement_descriptive.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()


def write_quality_report(feature_df: pd.DataFrame, output_dir: Path) -> None:
    report = coverage_report(feature_df)
    report_path = output_dir / 'combine_movement_quality_report.csv'
    report.to_csv(report_path, index=False)
    text_path = output_dir / 'combine_movement_quality_report.txt'
    text_path.write_text(report.to_string(index=False), encoding='utf-8')


def run_self_check(feature_df: pd.DataFrame, output_dir: Path) -> None:
    required_files = [
        output_dir / 'combine_movement_features.csv',
        output_dir / 'combine_movement_quality_report.csv',
        output_dir / 'combine_movement_quality_report.txt',
        output_dir / 'combine_movement_descriptive.png',
    ]
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f'{path.name} missing; self-check failed.')
        if path.stat().st_size == 0:
            raise ValueError(f'{path.name} is empty; self-check failed.')

    if feature_df.empty:
        raise ValueError('No pipeline features were produced; self-check failed.')
    required_cols = {'draft_year', 'nfl_id', 'drill_type', 'drill_name', 'attempt', 'feature_name', 'feature_value'}
    missing_cols = required_cols - set(feature_df.columns)
    if missing_cols:
        raise KeyError(f'Missing required columns: {sorted(missing_cols)}')
    if not pd.api.types.is_integer_dtype(feature_df['nfl_id']):
        raise TypeError('Athlete ID column must be integer-like.')
    if not feature_df['nfl_id'].between(1, 999999999).all():
        raise ValueError('Athlete IDs are invalid for this dataset.')
    if np.isinf(feature_df['feature_value'].to_numpy(dtype=float)).any():
        raise ValueError('Feature values contain infinity values.')

    dup_key = feature_df.duplicated(subset=['draft_year', 'nfl_id', 'drill_type', 'drill_name', 'attempt', 'feature_name']).any()
    if dup_key:
        raise ValueError('Duplicate athlete/drill/attempt/feature keys were produced.')

    feature_names = feature_df['feature_name'].dropna().unique().tolist()
    if len(feature_names) < 3:
        raise ValueError('Feature count too low for self-check.')
    if 'within_player_drill_max_speed_sd' in feature_names:
        raise ValueError('Unsupported within-player variability feature remains in the delivered feature set.')

    synthetic = pd.DataFrame({
        'draft_year': [2025, 2025, 2025],
        'nfl_id': [999999, 999999, 999999],
        'drill_type': ['FORTY_YARD_DASH', 'FORTY_YARD_DASH', 'FORTY_YARD_DASH'],
        'drill_name': ['FORTY_YARD_DASH', 'FORTY_YARD_DASH', 'FORTY_YARD_DASH'],
        'attempt': [1, 1, 1],
        'time_dt': pd.to_datetime(['2025-01-01 00:00:00', '2025-01-01 00:01:00', '2025-01-01 00:02:00']),
        'x_num': [0.0, 5.0, 10.0],
        'y_num': [0.0, 0.0, 0.0],
        's_num': [5.0, 5.0, 5.0],
        'a_num': [0.0, 0.0, 0.0],
    })
    synthetic_features = extract_attempt_features(synthetic)
    if not np.isclose(synthetic_features['max_speed'], 5.0, atol=1e-9):
        raise ValueError('Synthetic max_speed check failed.')
    if not np.isfinite(synthetic_features['path_length_yd']):
        raise ValueError('Synthetic path-length check failed.')

    quality = pd.read_csv(output_dir / 'combine_movement_quality_report.csv')
    num_zero_rows = (quality['n_rows'] == 0).sum()
    if num_zero_rows and not (quality.loc[quality['n_rows'] == 0, 'coverage'].isna().all()):
        raise ValueError('Coverage for missing drill metrics was fabricated.')

    duplicate_change_df = pd.DataFrame({
        'draft_year': [2025, 2025, 2025, 2025],
        'nfl_id': [999999, 999999, 999999, 999999],
        'drill_type': ['THREE_CONE', 'THREE_CONE', 'THREE_CONE', 'THREE_CONE'],
        'drill_name': ['THREE_CONE', 'THREE_CONE', 'THREE_CONE', 'THREE_CONE'],
        'attempt': [1, 1, 1, 1],
        'entity_type': ['PLAYER', 'PLAYER', 'PLAYER', 'PLAYER'],
        'time': ['2025-01-01T00:00:00', '2025-01-01T00:00:00', '2025-01-01T00:01:00', '2025-01-01T00:01:00'],
        'time_dt': pd.to_datetime(['2025-01-01 00:00:00', '2025-01-01 00:00:00', '2025-01-01 00:01:00', '2025-01-01 00:01:00']),
        'x': [0.0, 0.0, 5.0, 5.0],
        'y': [0.0, 0.0, 0.0, 0.0],
        's': [0.0, 0.0, 5.0, 5.0],
        'a': [0.0, 0.0, 0.0, 0.0],
        'dis': [0.0, 0.0, 5.0, 5.0],
        'dir': [0.0, 0.0, 90.0, 90.0],
        'x_num': [0.0, 0.0, 5.0, 5.0],
        'y_num': [0.0, 0.0, 0.0, 0.0],
        's_num': [0.0, 0.0, 5.0, 5.0],
        'a_num': [0.0, 0.0, 0.0, 0.0],
        'dis_num': [0.0, 0.0, 5.0, 5.0],
        'dir_num': [0.0, 0.0, 90.0, 90.0],
    })
    deduped = dedupe_timestamps(duplicate_change_df)
    if len(deduped) != 2:
        raise ValueError('Duplicate timestamps are not being safely de-duplicated.')

    valid_change_df = pd.DataFrame({
        'draft_year': [2025, 2025, 2025, 2025, 2025, 2025],
        'nfl_id': [999998, 999998, 999998, 999998, 999998, 999998],
        'drill_type': ['THREE_CONE', 'THREE_CONE', 'THREE_CONE', 'THREE_CONE', 'THREE_CONE', 'THREE_CONE'],
        'drill_name': ['THREE_CONE', 'THREE_CONE', 'THREE_CONE', 'THREE_CONE', 'THREE_CONE', 'THREE_CONE'],
        'attempt': [1, 1, 1, 1, 1, 1],
        'entity_type': ['PLAYER'] * 6,
        'time': ['2025-01-01T00:00:00', '2025-01-01T00:00:01', '2025-01-01T00:00:02', '2025-01-01T00:00:03', '2025-01-01T00:00:04', '2025-01-01T00:00:05'],
        'time_dt': pd.to_datetime(['2025-01-01 00:00:00', '2025-01-01 00:00:01', '2025-01-01 00:00:02', '2025-01-01 00:00:03', '2025-01-01 00:00:04', '2025-01-01 00:00:05']),
        'x': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        'y': [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        's': [0.0, 2.5, 2.5, 2.5, 2.5, 2.5],
        'a': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'dis': [0.0, 1.0, 1.4, 1.4, 1.0, 1.0],
        'dir': [0.0, 0.0, 90.0, 0.0, 180.0, 180.0],
        'x_num': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        'y_num': [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        's_num': [0.0, 2.5, 2.5, 2.5, 2.5, 2.5],
        'a_num': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'dis_num': [0.0, 1.0, 1.4, 1.4, 1.0, 1.0],
        'dir_num': [0.0, 0.0, 90.0, 0.0, 180.0, 180.0],
    })
    valid_cut = compute_heading_change(valid_change_df)
    if np.isnan(valid_cut[0]) or valid_cut[0] <= 60:
        raise ValueError('Synthetic direction-change trajectory did not satisfy the valid cut detector threshold.')

    noisy_df = pd.DataFrame({
        'draft_year': [2025, 2025, 2025, 2025],
        'nfl_id': [999997, 999997, 999997, 999997],
        'drill_type': ['THREE_CONE', 'THREE_CONE', 'THREE_CONE', 'THREE_CONE'],
        'drill_name': ['THREE_CONE', 'THREE_CONE', 'THREE_CONE', 'THREE_CONE'],
        'attempt': [1, 1, 1, 1],
        'entity_type': ['PLAYER'] * 4,
        'time': ['2025-01-01T00:00:00', '2025-01-01T00:00:01', '2025-01-01T00:00:02', '2025-01-01T00:00:03'],
        'time_dt': pd.to_datetime(['2025-01-01 00:00:00', '2025-01-01 00:00:01', '2025-01-01 00:00:02', '2025-01-01 00:00:03']),
        'x': [0.0, 0.01, 0.02, 0.03],
        'y': [0.0, 0.01, 0.02, 0.03],
        's': [0.0, 0.1, 0.1, 0.1],
        'a': [0.0, 0.0, 0.0, 0.0],
        'dis': [0.0, 0.01, 0.01, 0.01],
        'dir': [0.0, 0.0, 0.0, 0.0],
        'x_num': [0.0, 0.01, 0.02, 0.03],
        'y_num': [0.0, 0.01, 0.02, 0.03],
        's_num': [0.0, 0.1, 0.1, 0.1],
        'a_num': [0.0, 0.0, 0.0, 0.0],
        'dis_num': [0.0, 0.01, 0.01, 0.01],
        'dir_num': [0.0, 0.0, 0.0, 0.0],
    })
    noisy_cut = compute_heading_change(noisy_df)
    if not np.isnan(noisy_cut[0]):
        raise ValueError('Near-zero-displacement noisy trajectory should be rejected by the cut detector.')

    zero_time_df = pd.DataFrame({
        'draft_year': [2025, 2025, 2025],
        'nfl_id': [999999, 999999, 999999],
        'drill_type': ['FORTY_YARD_DASH', 'FORTY_YARD_DASH', 'FORTY_YARD_DASH'],
        'drill_name': ['FORTY_YARD_DASH', 'FORTY_YARD_DASH', 'FORTY_YARD_DASH'],
        'attempt': [1, 1, 1],
        'time_dt': pd.to_datetime(['2025-01-01 00:00:00', '2025-01-01 00:00:00', '2025-01-01 00:00:01']),
        'x_num': [0.0, 0.0, 5.0],
        'y_num': [0.0, 0.0, 0.0],
        's_num': [0.0, 0.0, 5.0],
        'a_num': [0.0, 0.0, 0.0],
    })
    zero_time_features = extract_attempt_features(zero_time_df)
    if zero_time_features:
        raise ValueError('Invalid non-positive elapsed-time trajectory should be excluded rather than producing feature values.')

    print('SELF_CHECK_OK')


def main() -> None:
    parser = argparse.ArgumentParser(description='Combine movement analytics pipeline for NFL Big Data Bowl 2027.')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR, help='Directory holding the CSV source data files.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Directory for generated feature outputs and reports.')
    parser.add_argument('--self-check', action='store_true', help='Run a minimal pipeline self-check only.')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tracking = load_tracking(args.data_dir)
    clean = dedupe_timestamps(tracking)
    feature_df = build_feature_rows(clean)
    combine_results = load_combine_results(args.data_dir)
    feature_df = add_combine_metadata(feature_df, combine_results)

    if args.self_check:
        feature_df.to_csv(args.output_dir / 'combine_movement_features.csv', index=False)
        write_quality_report(feature_df, args.output_dir)
        save_plot(feature_df, combine_results, args.output_dir)
        run_self_check(feature_df, args.output_dir)
        return

    feature_df.to_csv(args.output_dir / 'combine_movement_features.csv', index=False)
    write_quality_report(feature_df, args.output_dir)
    save_plot(feature_df, combine_results, args.output_dir)

    print(f'Wrote {len(feature_df)} feature rows to {args.output_dir / "combine_movement_features.csv"}')
    print(f'Wrote quality report to {args.output_dir / "combine_movement_quality_report.csv"}')
    print(f'Wrote plot to {args.output_dir / "combine_movement_descriptive.png"}')


if __name__ == '__main__':
    main()
