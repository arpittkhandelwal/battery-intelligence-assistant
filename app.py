"""
app.py
------
Streamlit dashboard for the Battery Time-Series Intelligence Assistant.

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd

from loader import load_battery_data
from preprocess import clean_data, detect_current_events
from features import compute_features, get_top_events, compute_window_features
from plot import (
    plotly_voltage_current,
    plotly_temperature,
    plotly_events_scatter,
    plotly_event_zoom,
    plotly_feature_heatmap,
)
import qa

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Battery Intelligence Assistant",
    page_icon="🔋",
    layout="wide",
)

DEFAULT_PATH = "/Users/arpitkhandelwal/Downloads/Research Project/Dataset Nasa.csv"

# ---------------------------------------------------------------------------
# Sidebar — data source selection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔋 Battery Intelligence")
    st.caption("Time-Series Analysis System")
    st.divider()

    source = st.radio("Data source", ["Default NASA dataset", "Upload CSV"])

    if source == "Default NASA dataset":
        dataset_path = DEFAULT_PATH
        uploaded = None
    else:
        uploaded = st.file_uploader("Upload battery CSV", type="csv")
        dataset_path = None

    threshold = st.slider(
        "Event detection threshold |di/dt|",
        min_value=0.01, max_value=0.5, value=0.05, step=0.01,
        help="Higher = only strong current spikes are marked."
    )
    st.divider()
    st.caption("Built with pandas · plotly · streamlit")


# ---------------------------------------------------------------------------
# Pipeline — cached so it only re-runs when inputs change
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and processing data…")
def run_pipeline(path_or_bytes, threshold, is_upload=False):
    """Run the full analysis pipeline and return (df, features, top_events)."""
    if is_upload:
        import io
        raw = pd.read_csv(io.BytesIO(path_or_bytes))
        # Write to temp file so loader can normalize it properly
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        raw.to_csv(tmp.name, index=False)
        tmp.close()
        df = load_battery_data(tmp.name)
        os.unlink(tmp.name)
    else:
        df = load_battery_data(path_or_bytes)

    df = clean_data(df)
    df = detect_current_events(df, threshold=threshold)
    features = compute_features(df)
    window_feats = compute_window_features(df, n_windows=20)
    top = get_top_events(df, n=10)
    return df, features, window_feats, top


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
try:
    if source == "Upload CSV" and uploaded is not None:
        df, features, window_feats, top_events = run_pipeline(uploaded.read(), threshold, is_upload=True)
    elif source == "Default NASA dataset":
        df, features, window_feats, top_events = run_pipeline(DEFAULT_PATH, threshold, is_upload=False)
    else:
        st.info("Upload a CSV file or switch to the default dataset to get started.")
        st.stop()
except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Battery Time-Series Intelligence Assistant")
st.caption(f"Analyzing: **{df['cell_id'].nunique()}** cell(s) · "
           f"**{len(df):,}** data points · "
           f"**{int(df['event_id'].dropna().nunique())}** events detected")

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Mean Voltage", f"{df['voltage'].mean():.3f} V")
k2.metric("Peak Current", f"{df['current'].max():.2f} A")
k3.metric("Max Temperature", f"{df['temperature'].max():.1f} °C")
k4.metric("Events Detected", int(df['event_id'].dropna().nunique()))

st.divider()

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_analysis, tab_events, tab_features, tab_bot = st.tabs([
    "📊 Analytics", "⚡ Events", "🔍 Features", "🤖 Assistant"
])

# ── Tab 1: Analytics ────────────────────────────────────────────────────────
with tab_analysis:
    st.subheader("Voltage & Current — Full Timeline")
    st.plotly_chart(plotly_voltage_current(df), use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Thermal Profile")
        st.plotly_chart(plotly_temperature(df), use_container_width=True)
    with col_r:
        st.subheader("Voltage Distribution")
        import plotly.express as px
        fig_hist = px.histogram(
            df, x='voltage', nbins=60,
            template="plotly_dark", color_discrete_sequence=["#4fc3f7"]
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# ── Tab 2: Events ────────────────────────────────────────────────────────────
with tab_events:
    st.subheader("Detected Current Events — Scatter by Intensity")
    st.plotly_chart(plotly_events_scatter(df), use_container_width=True)

    st.subheader("Top Events by |di/dt|")
    if not top_events.empty:
        st.dataframe(
            top_events.rename(columns={
                'event_id': 'Event #',
                'time': 'Time (s)',
                'di_dt': '|di/dt| (A/s)',
                'cell_id': 'Cell'
            }).reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Zoom into an Event")
        event_num = st.selectbox(
            "Select event to inspect",
            options=sorted(top_events['event_id'].astype(int).tolist())
        )
        st.plotly_chart(plotly_event_zoom(df, float(event_num)), use_container_width=True)
    else:
        st.info("No events detected at the current threshold. Try lowering it in the sidebar.")

# ── Tab 3: Features ──────────────────────────────────────────────────────────
with tab_features:
    import plotly.express as px
    import plotly.graph_objects as go

    # ── Overall summary card ────────────────────────────────────────────────
    st.subheader("Global Cell Summary")
    fmt_features = features.copy()
    fmt_features.columns = [
        'Voltage Drop (V)', 'Mean Current (A)', 'Temp Range (°C)',
        'Max |di/dt| (A/s)', 'Relaxation ΔV (V)', 'Degradation Slope (V/s)'
    ]
    st.dataframe(
        fmt_features.style
            .format({
                'Voltage Drop (V)':      '{:.4f}',
                'Mean Current (A)':      '{:.4f}',
                'Temp Range (°C)':       '{:.4f}',
                'Max |di/dt| (A/s)':     '{:.4f}',
                'Relaxation ΔV (V)':     '{:.5f}',
                'Degradation Slope (V/s)': '{:.3e}',
            })
            .background_gradient(cmap='RdYlGn', axis=0),
        use_container_width=True,
    )

    st.divider()

    # ── Window-level feature evolution ─────────────────────────────────────
    st.subheader("Feature Evolution Across 20 Time Windows")
    st.caption(
        "The recording is split into 20 equal time segments. "
        "Each row shows the computed features for that segment, "
        "revealing how battery state changes over the session."
    )

    # Heatmap: windows × numeric features
    numeric_cols = [
        'V mean (V)', 'V drop (V)', 'I mean (A)',
        'T mean (°C)', 'T range (°C)', '|di/dt| max', 'V slope (V/s)'
    ]
    heat_data = window_feats[numeric_cols].copy()
    # Normalize each column to [0, 1] for visual comparison
    heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min() + 1e-12)
    heat_norm.index = [f"W{int(w)}" for w in window_feats['window']]

    fig_heat = go.Figure(go.Heatmap(
        z=heat_norm.values,
        x=heat_norm.columns.tolist(),
        y=heat_norm.index.tolist(),
        colorscale='RdYlGn',
        reversescale=False,
        hovertemplate=(
            "Window: %{y}<br>"
            "Feature: %{x}<br>"
            "Normalised: %{z:.3f}<extra></extra>"
        ),
        colorbar=dict(title="Normalised"),
    ))
    fig_heat.update_layout(
        xaxis_title="Feature",
        yaxis_title="Time Window",
        height=520,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=20, t=20, b=60),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # ── Individual metric time-series ───────────────────────────────────────
    st.subheader("Metric Trends Over Time")
    metric_col, _ = st.columns([1, 2])
    with metric_col:
        selected_metric = st.selectbox(
            "Choose metric to plot",
            options=['V mean (V)', 'V drop (V)', 'I mean (A)',
                     'T mean (°C)', 'T range (°C)', '|di/dt| max', 'V slope (V/s)'],
        )
    fig_trend = px.line(
        window_feats,
        x='t_start (s)',
        y=selected_metric,
        color='cell_id',
        markers=True,
        template="plotly_dark",
        labels={'t_start (s)': 'Time (s)', selected_metric: selected_metric},
    )
    fig_trend.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.divider()

    # ── Raw window table ────────────────────────────────────────────────────
    with st.expander("📋 Raw window feature table"):
        st.dataframe(
            window_feats.drop(columns='cell_id')
                .set_index('window')
                .style.format('{:.4f}').background_gradient(cmap='Blues', axis=0),
            use_container_width=True,
        )

# ── Tab 4: Assistant ─────────────────────────────────────────────────────────
with tab_bot:
    st.subheader("Ask the Assistant")
    st.caption("Natural language queries routed to the analysis pipeline.")

    # Seed welcome message once
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                "Hello. I've analyzed the battery telemetry. Here are some things you can ask:\n\n"
                "- *Which cell has the largest voltage drop?*\n"
                "- *Show the degradation trend*\n"
                "- *Give me the top events*\n"
                "- *Summary report*"
            )
        }]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the battery data…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = qa.answer(prompt, df)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
