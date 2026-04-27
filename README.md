# Battery Time-Series Intelligence Assistant
## Technical Architecture & Research Report

**Project Context:** A production-quality, modular Python pipeline for automated ingestion, preprocessing, event detection, and feature extraction of lithium-ion battery telemetry, complete with a natural language Telegram Bot interface ([@VoltWatchhBot](https://t.me/VoltWatchhBot)).

---

## 1. System Architecture
The system is built on a decoupled, pipeline-oriented architecture. This ensures that analytical logic remains entirely independent of the deployment interface (CLI, Streamlit, or Telegram).

### 1.1 Core Research Insight

A key design decision in this system is to move away from purely statistical analysis of battery time-series data and instead focus on event-driven dynamics.

In real-world battery operation, critical degradation signals often emerge during short transient events, such as rapid current changes, rather than during steady-state operation. Traditional pipelines that rely only on global statistics (mean, variance, trends) may fail to capture these localized effects.

To address this, the system explicitly models current-driven transients by computing the temporal derivative of current ($di/dt$) and identifying high-intensity events. These events are then used as anchors for downstream analysis, including voltage response and thermal behavior.

This event-centric perspective enables:
* More realistic modeling of battery stress conditions
* Better interpretability of degradation indicators
* A foundation for future physics-informed or hybrid ML models

---

### 1.2 Core Modules
- **`loader.py` (Ingestion Engine):** 
  Handles the ingestion of raw CSV telemetry. Implements a robust `COLUMN_ALIASES` mapping system to automatically normalize diverse dataset headers (e.g., standardizing `voltage_measured`, `V`, `Voltage` to `voltage`).
- **`preprocess.py` (Signal Processing):** 
  Enforces a uniform time-step (1-second) via interpolation to resolve irregular sampling rates. Computes the discrete temporal derivative (`di/dt`) to detect high-intensity transient current events, using time-gap thresholds to group continuous spikes into discrete *events*.
- **`features.py` (Feature Extraction):** 
  Aggregates physics-based telemetry metrics. It computes global characteristics (Voltage Drop, Thermal Range, Relaxation $\Delta V$) and windowed metrics (time-evolution of degradation slope over 20 discrete segments).
- **`qa.py` (NLP Intent Router):** 
  A lightweight, rule-based inference engine. Uses keyword cluster scoring and phrase-length tiebreaking to map free-text user queries to internal Python functions, avoiding the latency and dependency overhead of external language models for simple queries.
- **`plot.py` (Visualization Backend):** 
  Generates dark-themed, publication-ready dual-axis charts (Voltage & Current over time). 
- **`bot.py` (Telegram Delivery Edge):** 
  An asynchronous `python-telegram-bot` deployment that interfaces directly with the QA engine, allowing users to query data and request zoomed-in matplotlib charts directly from their mobile devices.

---

## 2. Core Assumptions
During the development of the event detection and feature extraction engines, several domain-specific assumptions were formalized:

1. **Monotonic Time:** The dataset is assumed to represent a continuous chronological recording. Backwards-flowing timestamps are treated as fatal data corruption.
2. **Current Sign Agnosticism:** The thresholding engine takes the absolute derivative ($|di/dt|$) for event detection. We assume both rapid charging spikes (positive) and rapid load draws (negative) are equally important transient events.
3. **Linear Local Degradation:** The `V slope` feature utilizes a 1st-degree polynomial fit (linear regression) within local time windows, assuming degradation trends can be approximated linearly over short spans.
4. **Thermal Uniformity:** The telemetry records a single `temperature` series per cell, assuming this accurately reflects the internal core temperature rather than just external casing temperature.

---

## 3. Current Limitations
- **Cycle Segmentation:** The current pipeline analyzes the recording as a single continuous session. True battery health analysis (State of Health estimation) typically requires splitting data into distinct Charge/Discharge/Rest cycles to measure capacity fade ($Ah$) per cycle.
- **Rule-Based Fragility:** The NLP engine (`qa.py`) is extremely fast but fundamentally rigid. If a user asks *"Which battery got the most heated up?"*, the system relies on the word *"heat"* being explicitly hardcoded in the `temperature` intent array. It lacks semantic awareness.
- **Memory Bound:** The `loader.py` loads the entire dataset into a Pandas DataFrame in RAM. While fine for the NASA dataset (3,240 rows), scaling to gigabyte-scale high-frequency vehicle telemetry would require migrating to lazy-loading tools like Polars or Dask.

---

## 4. Future Roadmap: LLM & RAG Integration
The most powerful extension of this tool is upgrading the rigid `qa.py` router into an intelligent Agentic architecture using **Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)**.

### Phase 1: Semantic Intent Routing (LLM)
Replace the keyword-scoring logic with a lightweight, fast LLM (e.g., Gemini 1.5 Flash). 
- **Mechanism:** Pass the user's prompt to the LLM with a strict System Prompt: *"Map this query to one of the following tools: `[plot, stats, top_events, degradation]`"*.
- **Benefit:** Achieves 100% semantic flexibility. The user can ask in any language, using any phrasing (e.g., *"Show me a picture of the worst power spike"* → `plot_event`).

### Phase 2: RAG for Diagnostics
Battery degradation analysis often requires referencing external literature (e.g., identifying if a 1.2V drop signifies Lithium Plating or SEI layer growth).
- **Mechanism:** Ingest battery research papers, NASA documentation, and data sheets into a Vector Database (e.g., ChromaDB/FAISS).
- **Execution:** When the user asks *"Is a temperature range of 4.8°C normal during a 2A pulse?"*, the system extracts the telemetry data, retrieves matching thresholds from the RAG vector store, and synthesizes a highly grounded, scientifically accurate response.

### Phase 3: Function Calling (Tool Use)
Rather than hardcoding specific SQL/Pandas queries, we provide an LLM with the schema of the DataFrame and grant it a Python execution sandbox (or strict SQL bridge). The LLM autonomously writes the Pandas code to answer incredibly specific edge-case queries (e.g., *"What was the median temperature exactly 3 seconds after event #4?"*).
