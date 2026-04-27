"""
qa.py
-----
Rule-based question answering system for battery telemetry.

Design principle: no external NLP libraries needed.
Intent is detected by scoring keyword clusters against the user's input.
Entity extraction picks out cell IDs and numeric quantities directly.
"""

import re
import pandas as pd
import numpy as np
from typing import Optional
from features import compute_features, get_top_events


# ---------------------------------------------------------------------------
# Intent keyword registry
# ---------------------------------------------------------------------------
INTENTS = {
    # Generic event count — only fires on very specific phrasing
    'events':       ['how many event', 'count event', 'number of event', 'how many pulse'],
    # Top-events ranking — longer/more specific phrases score higher
    'top_events':   ['top event', 'give me the top', 'list event', 'show event',
                     'strongest event', 'biggest spike', 'rank event', 'worst pulse',
                     'most intense', 'all event', 'which event', 'best event'],
    'voltage_drop': ['voltage drop', 'delta v', 'v drop', 'drop in voltage', 'largest drop'],
    'temperature':  ['temperature', 'temp', 'thermal', 'hot', 'heat', 'warmest'],
    'degradation':  ['degradation', 'aging', 'health', 'soh', 'trend', 'fade'],
    'summary':      ['summary', 'overview', 'report', 'describe', 'tell me about'],
    'stats':        ['stat', 'average', 'mean', 'max', 'min', 'global', 'overall'],
    'plot':         ['plot', 'show', 'visualize', 'chart', 'graph', 'display'],
}


def _score_intent(question: str) -> str:
    """
    Score each intent against the lowercased question.
    Tiebreaker: longer matching keyword wins (more specific phrase beats a short one).
    """
    q = question.lower()
    scores = {intent: 0 for intent in INTENTS}
    max_kw_len = {intent: 0 for intent in INTENTS}

    for intent, keywords in INTENTS.items():
        for kw in keywords:
            if kw in q:
                scores[intent] += 1
                max_kw_len[intent] = max(max_kw_len[intent], len(kw))

    # Primary sort: score descending; tiebreaker: longest matched keyword descending
    best = max(scores, key=lambda i: (scores[i], max_kw_len[i]))
    return best if scores[best] > 0 else 'unknown'


def _extract_cell(question: str, available_cells) -> Optional[str]:
    """Check if any known cell ID appears verbatim in the question."""
    q = question.lower()
    for cell in available_cells:
        if str(cell).lower() in q:
            return cell
    return None


# ---------------------------------------------------------------------------
# Answer functions
# ---------------------------------------------------------------------------

def answer(question: str, df: pd.DataFrame) -> str:
    """
    Main entry point: parse the question and route to the right analysis.

    Args:
        question: Natural language user query.
        df: Preprocessed telemetry DataFrame.

    Returns:
        A formatted string answer.
    """
    features = compute_features(df)
    intent = _score_intent(question)
    target_cell = _extract_cell(question, features.index.tolist())

    # --- Voltage Drop ---
    if intent == 'voltage_drop':
        if target_cell:
            val = features.at[target_cell, 'voltage_drop']
            return f"**Voltage drop** for cell `{target_cell}`: **{val:.4f} V**"
        worst = features['voltage_drop'].idxmax()
        val = features.at[worst, 'voltage_drop']
        return (
            f"Cell with the **largest voltage drop**: `{worst}` — **{val:.4f} V**\n\n"
            f"This likely indicates higher internal resistance or impedance growth."
        )

    # --- Temperature ---
    elif intent == 'temperature':
        if target_cell:
            tr = features.at[target_cell, 'temp_range']
            return f"**Thermal range** for `{target_cell}`: **{tr:.2f} °C**"
        worst = features['temp_range'].idxmax()
        tr = features.at[worst, 'temp_range']
        return (
            f"Cell with **highest temperature variation**: `{worst}` — **{tr:.2f} °C**\n\n"
            f"Large thermal swings accelerate electrolyte decomposition."
        )

    # --- Degradation ---
    elif intent == 'degradation':
        worst = features['degradation_slope'].idxmin()
        slope = features.at[worst, 'degradation_slope']
        rows = []
        for cell, row in features.iterrows():
            status = "⚠️ Declining" if row['degradation_slope'] < -1e-6 else "✅ Stable"
            rows.append(f"| `{cell}` | {row['degradation_slope']:.3e} V/s | {status} |")
        table = "\n".join(["| Cell | V-trend (V/s) | Status |", "| :-- | :-- | :-- |"] + rows)
        return f"**Degradation Analysis** (voltage slope over time):\n\n{table}"

    # --- Summary ---
    elif intent == 'summary':
        cell = target_cell or features.index[0]
        f = features.loc[cell]
        status = "✅ Stable" if f['degradation_slope'] > -1e-6 else "⚠️ Degrading"
        return (
            f"### Diagnostic Report — `{cell}`\n\n"
            f"| Feature | Value |\n"
            f"| :-- | :-- |\n"
            f"| Voltage Drop | {f['voltage_drop']:.4f} V |\n"
            f"| Mean Current | {f['mean_current']:.3f} A |\n"
            f"| Temp Range | {f['temp_range']:.2f} °C |\n"
            f"| Relaxation ΔV | {f['relaxation_dv']:.4f} V |\n"
            f"| Max Event |di/dt| | {f['max_event_intensity']:.4f} A/s |\n"
            f"| Health Status | {status} |"
        )

    # --- Event count ---
    elif intent == 'events':
        if 'event_id' not in df.columns:
            return "Event detection has not been run. Re-run the pipeline with `detect_current_events`."
        n = int(df['event_id'].dropna().nunique())
        return f"**{n}** significant current-change events were detected in this dataset."

    # --- Global stats ---
    elif intent == 'stats':
        return (
            f"**Global Dataset Statistics**\n\n"
            f"| Metric | Value |\n| :-- | :-- |\n"
            f"| Mean Voltage | {df['voltage'].mean():.3f} V |\n"
            f"| Peak Current | {df['current'].max():.3f} A |\n"
            f"| Min Current | {df['current'].min():.3f} A |\n"
            f"| Max Temperature | {df['temperature'].max():.1f} °C |\n"
            f"| Total Rows | {len(df):,} |"
        )

    # --- Top events ---
    elif intent == 'top_events':
        top = get_top_events(df, n=16)   # show all detected events
        if top.empty:
            return "No events detected. Run `detect_current_events` first."
        n_total = int(df['event_id'].dropna().nunique()) if 'event_id' in df.columns else len(top)
        rows = [
            f"| {int(r['event_id'])} | {r['time']:.1f} | {r['di_dt']:.5f} | `{r['cell_id']}` |"
            for _, r in top.iterrows()
        ]
        table = "\n".join(
            ["| Event # | Time (s) | \u007cdi/dt\u007c (A/s) | Cell |",
             "| :-- | --: | --: | :-- |"] + rows
        )
        return (
            f"**{n_total} current-change events detected.** "
            f"Ranked by intensity (strongest first):\n\n{table}"
        )

    # --- Plot intent (Streamlit handles the actual figure) ---
    elif intent == 'plot':
        return (
            "📊 The **Analytics** tab contains interactive telemetry charts.\n\n"
            "You can zoom, pan, and inspect individual events there."
        )

    # --- Unknown ---
    else:
        return (
            "I didn't understand that query. You can ask about:\n\n"
            "- `voltage drop` — find cells with the highest V-drop\n"
            "- `temperature` — thermal variation across cells\n"
            "- `degradation` — health trend over time\n"
            "- `summary` — full diagnostic report per cell\n"
            "- `events` — count detected current pulses\n"
            "- `top events` — rank events by intensity\n"
            "- `stats` — overall dataset statistics"
        )
