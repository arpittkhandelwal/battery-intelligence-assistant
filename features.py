"""
features.py
-----------
Per-cell feature extraction from battery telemetry.

All features have physical interpretations:
  - voltage_drop: max observed V - min observed V (proxy for impedance rise)
  - mean_current: average discharge/charge rate
  - temp_range: max - min temperature (thermal stress indicator)
  - relaxation_dv: voltage change in 60s after each detected pulse end
  - max_event_intensity: strongest di/dt observed (event severity)
  - degradation_slope: linear trend of voltage over time (negative = aging)
"""

import numpy as np
import pandas as pd
from typing import Optional


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a summary feature table, one row per cell_id.

    Args:
        df: Preprocessed DataFrame (output of preprocess pipeline).

    Returns:
        Feature DataFrame indexed by cell_id.
    """
    records = []

    for cell_id, group in df.groupby('cell_id'):
        group = group.sort_values('time')
        record = {'cell_id': cell_id}

        # --- Voltage drop ---
        record['voltage_drop'] = group['voltage'].max() - group['voltage'].min()

        # --- Mean current ---
        record['mean_current'] = group['current'].mean()

        # --- Temperature range ---
        record['temp_range'] = group['temperature'].max() - group['temperature'].min()

        # --- Max current-change intensity (di/dt) ---
        if 'di_dt' in group.columns:
            record['max_event_intensity'] = group['di_dt'].max()
        else:
            record['max_event_intensity'] = np.nan

        # --- Voltage relaxation after pulse ---
        # Defined as the mean |dV/dt| in the 60 s following each event onset
        if 'event_id' in group.columns:
            record['relaxation_dv'] = _compute_relaxation(group)
        else:
            record['relaxation_dv'] = np.nan

        # --- Degradation slope (V vs t linear fit) ---
        # A strongly negative slope indicates rising internal resistance / capacity fade
        if len(group) > 1:
            slope, _ = np.polyfit(group['time'], group['voltage'], deg=1)
            record['degradation_slope'] = slope
        else:
            record['degradation_slope'] = np.nan

        records.append(record)

    feature_df = pd.DataFrame(records).set_index('cell_id')
    return feature_df


def _compute_relaxation(group: pd.DataFrame) -> float:
    """
    Estimate the average voltage relaxation amplitude following detected pulses.
    For each unique event, measure dV in the 60 s window after the event peak.
    """
    if 'event_id' not in group.columns:
        return np.nan

    event_ids = group['event_id'].dropna().unique()
    relaxations = []

    for eid in event_ids:
        event_time = group.loc[group['event_id'] == eid, 'time'].max()
        # Window: event end → event end + 60s
        window = group[(group['time'] > event_time) & (group['time'] <= event_time + 60)]
        if len(window) < 2:
            continue
        dv = window['voltage'].iloc[-1] - window['voltage'].iloc[0]
        relaxations.append(abs(dv))

    return float(np.mean(relaxations)) if relaxations else np.nan


def get_top_events(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Return the top-N events ranked by current-change intensity (di/dt).

    Args:
        df: DataFrame with di_dt and event_id columns.
        n: Number of top events to return.

    Returns:
        DataFrame with [event_id, time, di_dt, cell_id] for the top events.
    """
    if 'di_dt' not in df.columns or 'event_id' not in df.columns:
        return pd.DataFrame()

    events = df.dropna(subset=['event_id'])
    # For each event_id, take the row with the maximum di_dt
    idx_max = events.groupby('event_id')['di_dt'].idxmax()
    top = events.loc[idx_max, ['event_id', 'time', 'di_dt', 'cell_id']]
    top = top.sort_values('di_dt', ascending=False).head(n).reset_index(drop=True)
    return top


def compute_window_features(df: pd.DataFrame, n_windows: int = 20) -> pd.DataFrame:
    """
    Segment the recording into N equal time windows per cell and compute
    features for each window. This produces a rich multi-row table even
    when only one cell/file is loaded — showing how battery state evolves
    over the course of the recording.

    Args:
        df: Preprocessed DataFrame.
        n_windows: Number of windows to divide the timeline into.

    Returns:
        DataFrame with one row per (cell_id, window) and columns:
        [cell_id, window, t_start, t_end, voltage_mean, voltage_min,
         voltage_max, current_mean, temp_mean, temp_max, di_dt_max, slope]
    """
    records = []

    for cell_id, group in df.groupby('cell_id'):
        group = group.sort_values('time').reset_index(drop=True)
        t_min, t_max = group['time'].min(), group['time'].max()
        edges = np.linspace(t_min, t_max, n_windows + 1)

        for i in range(n_windows):
            t_start, t_end = edges[i], edges[i + 1]
            seg = group[(group['time'] >= t_start) & (group['time'] < t_end)]
            if len(seg) < 2:
                continue

            rec = {
                'cell_id':      cell_id,
                'window':       i + 1,
                't_start (s)':  round(t_start, 1),
                't_end (s)':    round(t_end, 1),
                'V mean (V)':   seg['voltage'].mean(),
                'V min (V)':    seg['voltage'].min(),
                'V max (V)':    seg['voltage'].max(),
                'V drop (V)':   seg['voltage'].max() - seg['voltage'].min(),
                'I mean (A)':   seg['current'].mean(),
                'T mean (°C)':  seg['temperature'].mean(),
                'T max (°C)':   seg['temperature'].max(),
                'T range (°C)': seg['temperature'].max() - seg['temperature'].min(),
            }

            # di/dt max within window
            if 'di_dt' in seg.columns:
                rec['|di/dt| max'] = seg['di_dt'].max()
            else:
                rec['|di/dt| max'] = np.nan

            # Local voltage slope (degradation proxy per window)
            slope, _ = np.polyfit(seg['time'], seg['voltage'], deg=1)
            rec['V slope (V/s)'] = slope

            records.append(rec)

    return pd.DataFrame(records)
