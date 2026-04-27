"""
plot.py
-------
All visualization functions for battery telemetry.
Uses matplotlib for CLI mode and plotly for the Streamlit dashboard.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Matplotlib (for CLI / standalone use)
# ---------------------------------------------------------------------------

def plot_voltage_current(df: pd.DataFrame, title: str = "Battery Telemetry") -> None:
    """
    Dual-axis plot of voltage and current over time.
    Highlights detected event rows with orange markers.
    """
    fig, ax1 = plt.subplots(figsize=(13, 5))

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Voltage (V)", color="#2980b9")
    ax1.plot(df['time'], df['voltage'], color="#2980b9", lw=1.4, label="Voltage")
    ax1.tick_params(axis='y', labelcolor="#2980b9")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Current (A)", color="#c0392b")
    ax2.plot(df['time'], df['current'], color="#c0392b", lw=0.9, alpha=0.5, label="Current")
    ax2.tick_params(axis='y', labelcolor="#c0392b")

    # Highlight detected events
    if 'is_event' in df.columns:
        events = df[df['is_event']]
        ax1.scatter(events['time'], events['voltage'],
                    color='orange', s=15, zorder=5, label='Current event')

    ax1.set_title(title)
    fig.tight_layout()
    plt.legend(loc='upper left')
    plt.show()


def plot_event_window(df: pd.DataFrame, event_id: float) -> None:
    """
    Zoom into a specific event window (matplotlib).
    """
    from preprocess import extract_event_window
    window = extract_event_window(df, event_id)
    if window.empty:
        print(f"[plot] No data found for event_id={event_id}")
        return
    plot_voltage_current(window, title=f"Event {int(event_id)} – Voltage / Current Response")


def plot_temperature(df: pd.DataFrame) -> None:
    """Plot temperature over time for each cell."""
    fig, ax = plt.subplots(figsize=(13, 4))
    for cell_id, group in df.groupby('cell_id'):
        ax.plot(group['time'], group['temperature'], lw=1.2, label=cell_id)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Thermal Profile")
    ax.legend()
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Plotly (for Streamlit dashboard – returns Figure objects)
# ---------------------------------------------------------------------------

_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)


def plotly_voltage_current(df: pd.DataFrame) -> go.Figure:
    """Interactive dual-axis voltage + current chart."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=df['time'], y=df['voltage'],
                   name="Voltage (V)", line=dict(color="#4fc3f7", width=1.5)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df['time'], y=df['current'],
                   name="Current (A)", line=dict(color="#ef5350", width=1), opacity=0.5),
        secondary_y=True,
    )

    # Mark events
    if 'is_event' in df.columns:
        ev = df[df['is_event']]
        fig.add_trace(
            go.Scatter(x=ev['time'], y=ev['voltage'],
                       mode='markers', name='Events',
                       marker=dict(color='orange', size=5, symbol='x')),
            secondary_y=False,
        )

    fig.update_yaxes(title_text="Voltage (V)", secondary_y=False,
                     gridcolor="#2d323e")
    fig.update_yaxes(title_text="Current (A)", secondary_y=True,
                     gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title_text="Time (s)", gridcolor="#2d323e")
    fig.update_layout(height=420, hovermode="x unified",
                      legend=dict(orientation="h", y=1.05), **_DARK)
    return fig


def plotly_temperature(df: pd.DataFrame) -> go.Figure:
    """Interactive thermal profile chart."""
    fig = go.Figure()
    for cell_id, group in df.groupby('cell_id'):
        fig.add_trace(go.Scatter(
            x=group['time'], y=group['temperature'],
            name=cell_id, line=dict(width=1.5)
        ))
    fig.update_layout(
        title="Thermal Profile",
        xaxis_title="Time (s)",
        yaxis_title="Temperature (°C)",
        height=320, **_DARK
    )
    return fig


def plotly_events_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of detected events coloured by di/dt intensity."""
    if 'event_id' not in df.columns:
        return go.Figure()
    ev = df.dropna(subset=['event_id'])
    fig = go.Figure(go.Scatter(
        x=ev['time'], y=ev['voltage'],
        mode='markers',
        marker=dict(
            color=ev['di_dt'],
            colorscale='Plasma',
            size=7,
            showscale=True,
            colorbar=dict(title="|di/dt|")
        ),
        text=ev['event_id'].astype(int).astype(str).apply(lambda e: f"Event {e}"),
        hovertemplate="%{text}<br>t=%{x:.1f}s<br>V=%{y:.3f}V<extra></extra>"
    ))
    fig.update_layout(
        title="Detected Current Events",
        xaxis_title="Time (s)",
        yaxis_title="Voltage (V)",
        height=340, **_DARK
    )
    return fig


def plotly_event_zoom(df: pd.DataFrame, event_id: float) -> go.Figure:
    """Zoom into a single event window (Plotly)."""
    from preprocess import extract_event_window
    window = extract_event_window(df, event_id)
    if window.empty:
        return go.Figure()
    return plotly_voltage_current(window)


def plotly_feature_heatmap(features: pd.DataFrame) -> go.Figure:
    """Heatmap of all extracted features per cell."""
    # Normalize each column to [0,1] for visual comparison
    norm = (features - features.min()) / (features.max() - features.min() + 1e-9)
    fig = go.Figure(go.Heatmap(
        z=norm.values,
        x=norm.columns.tolist(),
        y=norm.index.tolist(),
        colorscale='Viridis',
        hovertemplate="Cell: %{y}<br>Feature: %{x}<br>Value: %{z:.3f}<extra></extra>"
    ))
    fig.update_layout(
        title="Feature Heatmap (normalized)",
        height=300, **_DARK
    )
    return fig
