import numpy as np
import pandas as pd
from pathlib import Path

data_path = Path(r"D:\NFL\data\combine_tracking.csv")

def prep():
    df = pd.read_csv(data_path, low_memory=False)
    out = df[df['entity_type'].eq('PLAYER')].copy()
    out['time_dt'] = pd.to_datetime(out['time'], errors='coerce')
    out['x_num'] = pd.to_numeric(out['x'], errors='coerce')
    out['y_num'] = pd.to_numeric(out['y'], errors='coerce')
    out['s_num'] = pd.to_numeric(out['s'], errors='coerce')
    out['a_num'] = pd.to_numeric(out['a'], errors='coerce')
    out['dis_num'] = pd.to_numeric(out['dis'], errors='coerce')
    out['dir_num'] = pd.to_numeric(out['dir'], errors='coerce')
    out = out.sort_values(['nfl_id','drill_type','attempt','time_dt']).reset_index(drop=True)
    return out


def speed_from_xy(g, smooth=0):
    g = g.sort_values('time_dt').copy()
    t = (g['time_dt'].astype('int64') / 1_000_000_000).to_numpy(float)
    dt = np.diff(t)
    dt = np.clip(dt, 1e-6, None)
    x = g['x_num'].to_numpy(float)
    y = g['y_num'].to_numpy(float)
    dx = np.diff(x)
    dy = np.diff(y)
    disp = np.hypot(dx, dy)
    vel = disp / dt
    if smooth:
        kernel = np.ones(smooth) / smooth
        vel = np.convolve(vel, kernel, mode='same')
        vel = np.concatenate([[vel[0]], vel])
    return vel, dt, t


def signed_accel(g, smooth=0):
    v, dt, t = speed_from_xy(g, smooth=smooth)
    coeff = np.gradient(v, t[:len(v)]) if len(v) > 1 else np.zeros_like(v)
    return coeff


def detect_cut_idx(g, smooth=3):
    g = g.sort_values('time_dt').copy()
    x = g['x_num'].to_numpy(float)
    y = g['y_num'].to_numpy(float)
    if len(x) < 6:
        return []
    # smooth lightly in x and y and compute heading changes
    xr = pd.Series(x).rolling(smooth, center=True, min_periods=1).mean().to_numpy()
    yr = pd.Series(y).rolling(smooth, center=True, min_periods=1).mean().to_numpy()
    dx = np.diff(xr)
    dy = np.diff(yr)
    head = np.degrees(np.arctan2(dy, dx))
    head = (head + 360) % 360
    delta = np.abs((head[1:] - head[:-1] + 180) % 360 - 180)
    # require minimum angular change over a minimum duration
    cut_candidates = np.where((delta >= 25) & (np.r_[False, np.convolve(np.ones(3), (delta >= 25).astype(float), mode='same') > 1.5]))[0]
    return list(cut_candidates + 1)


def summarize_attempt(g, smooth=3):
    g = g.sort_values('time_dt').copy()
    t = (g['time_dt'].astype('int64') / 1_000_000_000).to_numpy(float)
    t_rel = t - t[0]
    x = g['x_num'].to_numpy(float)
    y = g['y_num'].to_numpy(float)
    s = g['s_num'].to_numpy(float)
    a = g['a_num'].to_numpy(float)
    v_xy, dt, _ = speed_from_xy(g, smooth=smooth)
    if len(v_xy) == 0:
        return None
    # fit to speed-like shape using raw trajectory speed
    # speed resemblance check
    speed_rmse = float(np.sqrt(np.mean((v_xy[:len(s)] - s) ** 2))) if len(v_xy) >= len(s) else np.nan
    # acceleration from smoothed speed
    acc = np.gradient(v_xy, t_rel[:len(v_xy)]) if len(v_xy) > 1 else np.zeros_like(v_xy)
    a_sm = acc
    # displacement checks
    dx = np.diff(x)
    dy = np.diff(y)
    path = np.hypot(dx, dy)
    dt2 = np.diff(t_rel)
    dt2 = np.clip(dt2, 1e-6, None)
    # anomaly detection
    bad = []
    if len(path) > 0:
        bad = np.where((path > 3) | (path / dt2 > 25))[0]
    cuts = detect_cut_idx(g, smooth=smooth)
    # candidate for summary values
    res = {
        'nfl_id': int(g['nfl_id'].iloc[0]),
        'drill_type': g['drill_type'].iloc[0],
        'drill_name': g['drill_name'].iloc[0],
        'attempt': int(g['attempt'].iloc[0]),
        'frames': len(g),
        'speed_rmse': speed_rmse,
        'acc_rmse': float(np.sqrt(np.mean((a_sm[1:len(a)] - a[1:]) ** 2))) if len(a) > 1 and len(a_sm) >= len(a) else np.nan,
        'dt_med': float(np.median(np.diff(t_rel))),
        'dt_min': float(np.min(np.diff(t_rel))),
        'dt_max': float(np.max(np.diff(t_rel))),
        'dis_anomalies': int(len(bad)),
        'cut_count': len(cuts),
        'x_range': (float(x.min()), float(x.max())),
        'y_range': (float(y.min()), float(y.max())),
        's_max': float(np.nanmax(s)) if len(s) else np.nan,
    }
    return res


def main():
    df = prep()
    print('Rows player-only', len(df))
    print('Unique drill types', sorted(df['drill_type'].unique().tolist()))
    # Revalidate s against x/y-derived velocity
    rows = []
    for key, g in df.groupby(['nfl_id','drill_type','attempt']):
        info = summarize_attempt(g, smooth=3)
        if info is not None:
            rows.append(info)
    qc = pd.DataFrame(rows)
    print('Speed RMSE median', qc['speed_rmse'].median())
    print('Speed RMSE p95', qc['speed_rmse'].quantile(0.95))
    print('Acceleration RMSE median', qc['acc_rmse'].median())
    print('Acceleration RMSE p95', qc['acc_rmse'].quantile(0.95))
    print('Dis anomaly counts by attempt summary')
    print(qc['dis_anomalies'].describe().to_string())
    print('Cut count summary')
    print(qc['cut_count'].describe().to_string())

    # Inspect 8,022 anomalous dis frames
    bad = []
    for key, g in df.groupby(['nfl_id','drill_type','attempt']):
        g = g.sort_values('time_dt').copy()
        t = (g['time_dt'].astype('int64') / 1_000_000_000).to_numpy(float)
        dt = np.diff(t)
        dx = np.diff(g['x_num'].to_numpy(float))
        dy = np.diff(g['y_num'].to_numpy(float))
        path = np.hypot(dx, dy)
        loc = np.where((path > 3) | (path / np.clip(dt, 1e-6, None) > 25))[0]
        if len(loc):
            for idx in loc:
                bad.append({
                    'nfl_id': key[0],
                    'drill': key[1],
                    'attempt': key[2],
                    'path': float(path[idx]),
                    'dt': float(dt[idx]),
                    's': float(g['s_num'].iloc[idx+1]),
                    'a': float(g['a_num'].iloc[idx+1]),
                    'x1': float(g['x_num'].iloc[idx]),
                    'x2': float(g['x_num'].iloc[idx+1]),
                    'y1': float(g['y_num'].iloc[idx]),
                    'y2': float(g['y_num'].iloc[idx+1]),
                    'dis': float(g['dis_num'].iloc[idx+1]),
                    'time': g['time'].iloc[idx+1],
                })
    bad_df = pd.DataFrame(bad)
    if not bad_df.empty:
        print('Bad displacement sample rows', len(bad_df))
        print(bad_df.head(10).to_string(index=False))
    else:
        print('No large displacement anomalies found')

    # sensitivity to smoothing parameters
    smooth_levels = [0, 1, 3, 5]
    rank_corrs = []
    for sm in smooth_levels:
        feat = []
        for key, g in df.groupby(['nfl_id','drill_type','attempt']):
            g = g.sort_values('time_dt').copy()
            s = g['s_num'].to_numpy(float)
            if len(s) < 3:
                continue
            vel, _, _ = speed_from_xy(g, smooth=sm)
            if len(vel) >= len(s):
                max_speed = float(np.max(s))
                feat.append((key[0], key[1], key[2], max_speed, float(np.max(vel[:len(s)]))))
        if feat:
            m = pd.DataFrame(feat, columns=['nfl_id','drill','attempt','s_raw','s_xy'])
            print('SMOOTH', sm, 'speed raw vs xy corr', m['s_raw'].corr(m['s_xy']))

    # drill-level summary for the candidate-family validation
    print('Candidate family viability (per drill, attempt-level)')
    for drill in ['FORTY_YARD_DASH','SHORT_SHUTTLE','THREE_CONE_DRILL','SKILL_DRILLS_DB','SKILL_DRILLS_OL','SKILL_DRILLS_WR']:
        g = df[df['drill_type'] == drill]
        print(drill, 'players', g['nfl_id'].nunique(), 'attempts', g.groupby(['nfl_id','attempt']).ngroups)


if __name__ == '__main__':
    main()
