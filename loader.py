"""
loader.py
---------
Handles loading and schema normalization of battery telemetry CSV files.
Supports single files, lists of files, or directories.
"""

import os
import glob
import pandas as pd
from typing import Union, List


# Canonical column mapping: any recognized alias → internal standard name
COLUMN_ALIASES = {
    'voltage_measured': 'voltage',
    'current_measured': 'current',
    'temperature_measured': 'temperature',
    'temp': 'temperature',
    'volt': 'voltage',
    'time': 'time',
    'voltage': 'voltage',
    'current': 'current',
    'temperature': 'temperature',
}

REQUIRED_COLUMNS = {'time', 'voltage', 'current', 'temperature'}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and lowercase column names, then apply alias mapping."""
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {c: COLUMN_ALIASES[c] for c in df.columns if c in COLUMN_ALIASES}
    return df.rename(columns=rename)


def _validate(df: pd.DataFrame, source: str) -> bool:
    """Check that all required columns exist after normalization."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        print(f"[loader] Skipping '{source}': missing columns {missing}")
        return False
    return True


def load_battery_data(path: Union[str, List[str]]) -> pd.DataFrame:
    """
    Load one or many CSV files into a single normalized DataFrame.

    Args:
        path: A file path, directory path, or list of file paths.

    Returns:
        Concatenated DataFrame with columns:
        [time, voltage, current, temperature, cell_id]
    """
    # Resolve file list
    if isinstance(path, list):
        files = path
    elif os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, '*.csv')))
    elif os.path.isfile(path):
        files = [path]
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    if not files:
        raise ValueError(f"No CSV files found at: {path}")

    frames = []
    for f in files:
        try:
            raw = pd.read_csv(f)
        except Exception as e:
            print(f"[loader] Could not read '{f}': {e}")
            continue

        if raw.empty:
            print(f"[loader] Skipping empty file: '{f}'")
            continue

        df = _normalize_columns(raw)

        if not _validate(df, f):
            continue

        # Assign cell identifier from the filename (without extension)
        df['cell_id'] = os.path.splitext(os.path.basename(f))[0]

        # Coerce all required columns to numeric (drop non-convertible rows)
        for col in REQUIRED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=list(REQUIRED_COLUMNS))

        frames.append(df[list(REQUIRED_COLUMNS) + ['cell_id']])

    if not frames:
        raise ValueError("No valid battery data found. Check file format and required columns.")

    result = pd.concat(frames, ignore_index=True)
    print(f"[loader] Loaded {len(result):,} rows from {len(frames)} file(s).")
    return result
