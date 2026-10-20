import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path(r"D:\NFL\data\combine_tracking.csv")


def clean_player_attempts(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df['entity_type'].eq('PLAYER')].copy()
    out['time_dt'] = pd.to_datetime(out['time'], errors='coerce')
    out['x_num'] = pd.to_numeric(out['x'], errors='coerce')
    out['y_num'] = pd.to_numeric(out['y'], errors='coerce')
    out['s_num'] = pd.to_numeric(out['s'], errors='coerce')
    out['a_num'] = pd.to_numeric(out['a'], errors='coerce')
    out['dis_num'] = pd.to_numeric(out['dis'], errors='coerce')
    out['dir_num'] = pd.to_numeric(out['dir'], errors='coerce')
    out = out.sort_values(['nfl_id', 'drill_type', 'attempt', 'time_dt']).reset_index(drop=True)
    return out


def attempt_metrics(g: pd.DataFrame):
    g = g.sort_values('time_dt').copy()
    if len(g) < 2:
        return None
    t = g['time_dt'].astype('int64') / 1_000_000_000
    dt = np.diff(t.to_numpy())
    dt = np.clip(dt, 1e-6, None)
    x = g['x_num'].to_numpy(float)
    y = g['y_num'].to_numpy(float)
    disp = np.hypot(np.diff(x), np.diff(y))
    speed_from_disp = disp / dt
    s = g['s_num'].to_numpy(float)
    s_valid = s[1:]
    s_err = speed_from_disp - s_valid
    speed_rmse = float(np.sqrt(np.mean(s_err ** 2)))
    a_est = np.gradient(s, t.to_numpy())
    a_err = a_est[1:] - g['a_num'].to_numpy(float)[1:]
    a_rmse = float(np.sqrt(np.mean(a_err[np.isfinite(a_err)] ** 2)))
    extreme_dis = g['dis_num'].sort_values(ascending=False).head(10).to_dict()
    dt_jitter = np.diff(t.to_numpy())
    res = {
        'nfl_id': g['nfl_id'].iloc[0],
        'drill_type': g['drill_type'].iloc[0],
        'drill_name': g['drill_name'].iloc[0],
        'attempt': int(g['attempt'].iloc[0]),
        'frames': len(g),
        'speed_rmse': speed_rmse,
        'a_rmse': a_rmse,
        'dt_median': float(np.median(dt)),
        'dt_p01': float(np.quantile(dt, 0.01)),
        'dt_p99': float(np.quantile(dt, 0.99)),
        'dir_to_motion_corr': None,
        'extreme_dis': extreme_dis,
        'x_range': (float(x.min()), float(x.max())),
        'y_range': (float(y.min()), float(y.max())),
        's_range': (float(s.min()), float(s.max())),
    }
    return res


def motion_heading_from_xy(g: pd.DataFrame):
    g = g.sort_values('time_dt').copy()
    if len(g) < 2:
        return None
    dx = np.diff(g['x_num'].to_numpy(float))
    dy = np.diff(g['y_num'].to_numpy(float))
    h = np.degrees(np.arctan2(dy, dx))
    h = (h + 360) % 360
    return h


def validate_dir_semantics(df: pd.DataFrame):
    # check whether dir approximates motion direction rather than body orientation
    # we will inspect a few attempts by correlation between |dir - heading|
    rows = []
    for key, g in df.groupby(['nfl_id', 'drill_type', 'attempt']):
        if len(g) < 2:
            continue
        heading = motion_heading_from_xy(g)
        dir_vals = pd.to_numeric(g['dir_num'], errors='coerce').to_numpy()[1:]
        if len(heading) != len(dir_vals):
            continue
        diff = np.abs((dir_vals - heading + 180) % 360 - 180)
        rows.append(float(np.median(diff)))
    if not rows:
        return None
    return np.median(rows), np.quantile(rows, 0.9), len(rows)


def compute_feature_summary(df: pd.DataFrame):
    summary = []
    for key, g in df.groupby(['nfl_id', 'drill_type', 'attempt']):
        g = g.sort_values('time_dt').copy()
        if len(g) < 3:
            continue
        s = g['s_num'].to_numpy(float)
        a = g['a_num'].to_numpy(float)
        x = g['x_num'].to_numpy(float)
        y = g['y_num'].to_numpy(float)
        t = (g['time_dt'].astype('int64') / 1_000_000_000).to_numpy(float)
        dt = np.diff(t)
        dt = np.clip(dt, 1e-6, None)
        pos = np.column_stack([x, y])
        dx = np.diff(x)
        dy = np.diff(y)
        path = np.hypot(dx, dy)
        path_length = float(path.sum())
        net_disp = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        eff = float(net_disp / path_length) if path_length > 0 else np.nan
        s_max = float(np.nanmax(s)) if np.isfinite(s).any() else np.nan
        tmax = float(t[np.nanargmax(s)]) if np.isfinite(s).any() else np.nan
        # 90% speed time, using first threshold crossing after start
        thr = 0.9 * s_max
        idx = np.where(s >= thr)[0]
        t90 = float(t[idx[0]]) if len(idx) else np.nan
        # acceleration impulse (positive acceleration integrated)
        pos_a = np.clip(a, 0, None)
        impulse = float(np.sum(pos_a * np.r_[dt[0], dt])) if len(dt) else np.nan
        # first 1s window early acceleration proxy: distance covered in first 1 sec, speed mean over first 1 sec
        mask = t - t[0] <= 1.0
        early_speed = float(np.mean(s[mask])) if np.any(mask) else np.nan
        early_distance = float(np.sum(path[(t[:-1] - t[0]) <= 1.0])) if len(path) else np.nan
        # braking proxy: find the post-peak decreasing segment
        if len(s) > 2:
            peak_idx = int(np.argmax(s))
            if peak_idx < len(s) - 1:
                dec_seg = s[peak_idx:]
                dec_min_idx = peak_idx + int(np.argmin(dec_seg))
                v_before = float(s[peak_idx])
                v_after = float(s[dec_min_idx])
                brake_duration = float(t[dec_min_idx] - t[peak_idx])
                brake_distance = float(np.sum(path[max(peak_idx - 1, 0):max(dec_min_idx - 1, 0)])) if dec_min_idx > peak_idx else 0.0
                max_dec = float(np.min(np.diff(s) / np.clip(np.diff(t), 1e-6, None)))
            else:
                v_before = v_after = brake_duration = brake_distance = max_dec = np.nan
        else:
            v_before = v_after = brake_duration = brake_distance = max_dec = np.nan
        # directional change proxy
        # determine a candidate turn if heading changes by >= 30 deg in one step and there is speed > 0.5
        heading = np.degrees(np.arctan2(np.diff(y), np.diff(x)))
        heading = (heading + 360) % 360
        if len(heading) > 1:
            delta_heading = np.abs(np.mod(heading[1:] - heading[:-1] + 180, 360) - 180)
            cut_idx = np.where((delta_heading > 30) & (s[1:-1] > 0.5))[0]
            if len(cut_idx):
                idx = int(cut_idx[0]) + 1
                speed_before = float(s[idx])
                speed_min = float(np.min(s[max(0, idx - 2): min(len(s), idx + 3)]))
                speed_after = float(s[idx + 1]) if idx + 1 < len(s) else float(s[idx])
                ret = float((speed_after / speed_before) * 100) if speed_before > 0 else np.nan
            else:
                speed_before = speed_min = speed_after = ret = np.nan
        else:
            speed_before = speed_min = speed_after = ret = np.nan
        # reacceleration metric: from min to 90% of pre-cut speed
        if np.isfinite(speed_before) and np.isfinite(speed_after):
            post_idx = np.where(s >= 0.9 * speed_before)[0]
            t_reacc = float(t[post_idx[0]] - t[idx]) if len(post_idx) and post_idx[0] > idx else np.nan
            a_post = float(a[idx + 1]) if idx + 1 < len(a) else np.nan
        else:
            t_reacc = a_post = np.nan
        summary.append({
            'nfl_id': int(key[0]),
            'drill_type': key[1],
            'attempt': int(key[2]),
            't90': t90,
            'tmax': tmax,
            'accel_impulse': impulse,
            'early_speed_1s': early_speed,
            'early_distance_1s': early_distance,
            'max_dec': max_dec,
            'brake_speed_start': v_before,
            'brake_duration': brake_duration,
            'brake_distance': brake_distance,
            'speed_before_cut': speed_before,
            'speed_min_after_cut': speed_min,
            'speed_after_cut': speed_after,
            'speed_retained_pct': ret,
            'reaccel_time_to_90pct': t_reacc,
            'reaccel_after_cut_a': a_post,
            'path_length': path_length,
            'net_displacement': net_disp,
            'path_efficiency': eff,
            'max_speed': s_max,
        })
    return pd.DataFrame(summary)


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    player_df = clean_player_attempts(df)
    print('PLAYER_ROWS', len(player_df))
    print('TOTAL_PLAYER_ATTEMPTS', player_df.groupby(['nfl_id', 'drill_type', 'attempt']).ngroups)
    # data quality validation
    qc = []
    for key, g in player_df.groupby(['nfl_id', 'drill_type', 'attempt']):
        m = attempt_metrics(g)
        if m is not None:
            qc.append(m)
    qc_df = pd.DataFrame(qc)
    if not qc_df.empty:
        print('SPEED_RMSE_MEDIAN', qc_df['speed_rmse'].median())
        print('A_RMSE_MEDIAN', qc_df['a_rmse'].median())
        print('SPEED_RMSE_P95', qc_df['speed_rmse'].quantile(0.95))
        print('A_RMSE_P95', qc_df['a_rmse'].quantile(0.95))
        print('MAX_DIS_VALUES')
        print(pd.concat([pd.DataFrame({'max_dis': player_df['dis_num'].sort_values(ascending=False).head(10).to_numpy()})]))
    print('DIR_SEMANTICS_CHECK', validate_dir_semantics(player_df))

    # inspect a few attempts that are manually inspectable
    sample_attempts = []
    for key, g in player_df.groupby(['nfl_id', 'drill_type', 'attempt']):
        if g['drill_type'].iloc[0] in {'FORTY_YARD_DASH', 'SHORT_SHUTTLE', 'THREE_CONE_DRILL'}:
            sample_attempts.append(key)
        if len(sample_attempts) >= 3:
            break

    print('SELF_CHECK_ATTEMPTS')
    for key in sample_attempts:
        g = player_df[(player_df['nfl_id'] == key[0]) & (player_df['drill_type'] == key[1]) & (player_df['attempt'] == key[2])].sort_values('time_dt')
        print('KEY', key)
        print(g[['time','x','y','s','a','dis','dir']].head(5).to_string(index=False))
        m = attempt_metrics(g)
        if m:
            print('METRICS', m)
        print('---')

    print('FEATURE_SUMMARY_FOR_FIRST_5_TRAJECTS')
    feat = compute_feature_summary(player_df)
    print(feat.head().to_string(index=False))

    # minimal aggregate summary for candidate features
    print('CANDIDATE_FEATURE_DISTRIBUTION')
    for col in ['t90', 'tmax', 'accel_impulse', 'early_speed_1s', 'max_dec', 'brake_duration', 'path_efficiency']:
        s = feat[col].dropna()
        if len(s):
            print(col, 'median=', float(s.median()), 'p05=', float(s.quantile(0.05)), 'p95=', float(s.quantile(0.95)), 'missing=', int(feat[col].isna().sum()))

    print('DONE')


if __name__ == '__main__':
    main()
