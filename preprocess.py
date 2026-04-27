"""
preprocess.py
-------------
Data cleaning, resampling, event detection, and window extraction.
All functions are pure (input → output), with no side effects.
"""

import numpy as np
import pandas as pd
from typing import Optional


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by [cell_id, time], interpolate internal NaNs, and
    forward/backward fill any remaining edge NaNs.
    """
    df = df.copy()
    df = df.sort_values(by=['cell_id', 'time']).reset_index(drop=True)

    # Interpolate per cell to avoid cross-cell contamination
    numeric_cols = ['voltage', 'current', 'temperature']
    df[numeric_cols] = (
        df.groupby('cell_id')[numeric_cols]
        .transform(lambda g: g.interpolate(method='linear').ffill().bfill())
    )
    return df


def resample_uniform(df: pd.DataFrame, interval_s: float = 1.0) -> pd.DataFrame:
    """
    Optional: Resample each cell to a uniform time grid.
    Useful for algorithms that assume equal spacing.

    Args:
        df: Cleaned DataFrame.
        interval_s: Desired time step in seconds.

    Returns:
        Uniformly-resampled DataFrame.
    """
    frames = []
    for cell_id, group in df.groupby('cell_id'):
        t_min = group['time'].min()
        t_max = group['time'].max()
        new_time = np.arange(t_min, t_max, interval_s)

        resampled = pd.DataFrame({'time': new_time})
        for col in ['voltage', 'current', 'temperature']:
            resampled[col] = np.interp(new_time, group['time'].values, group[col].values)
        resampled['cell_id'] = cell_id
        frames.append(resampled)

    return pd.concat(frames, ignore_index=True)


def detect_current_events(
    df: pd.DataFrame,
    threshold: float = 0.05,
    min_gap: int = 5
) -> pd.DataFrame:
    """
    Mark rows where the absolute current derivative (di/dt) exceeds a threshold.
    Adjacent events within `min_gap` rows are treated as one event.

    Args:
        df: Cleaned DataFrame.
        threshold: Minimum |di/dt| to qualify as an event.
        min_gap: Minimum row gap between distinct events.

    Returns:
        DataFrame with added columns:
          di_dt      — raw current derivative
          is_event   — boolean flag
          event_id   — integer event group (NaN for non-event rows)
    """
    df = df.copy()
    # Per-cell derivative to avoid jumps at cell boundaries
    df['di_dt'] = df.groupby('cell_id')['current'].diff().abs()
    df['di_dt'] = df['di_dt'].fillna(0)

    df['is_event'] = df['di_dt'] > threshold

    # Assign unique event IDs by grouping consecutive True rows
    event_id_series = pd.Series(np.nan, index=df.index, dtype=float)
    event_counter = 0
    last_event_idx = -min_gap - 1

    for idx in df.index[df['is_event']]:
        if idx - last_event_idx >= min_gap:
            event_counter += 1
            last_event_idx = idx
        event_id_series.at[idx] = float(event_counter)

    df['event_id'] = event_id_series
    return df


def extract_event_window(
    df: pd.DataFrame,
    event_id: float,
    pre_seconds: float = 10.0,
    post_seconds: float = 60.0
) -> pd.DataFrame:
    """
    Extract a time window around a specific event for detailed analysis.

    Args:
        df: DataFrame with event detection columns.
        event_id: The event to zoom into.
        pre_seconds: How many seconds before the event to include.
        post_seconds: How many seconds after the event to include.

    Returns:
        Sub-DataFrame covering the event window.
    """
    event_rows = df[df['event_id'] == event_id]
    if event_rows.empty:
        return pd.DataFrame()

    t_event = event_rows['time'].mean()
    cell = event_rows['cell_id'].iloc[0]

    mask = (
        (df['cell_id'] == cell) &
        (df['time'] >= t_event - pre_seconds) &
        (df['time'] <= t_event + post_seconds)
    )
    return df[mask].copy()
