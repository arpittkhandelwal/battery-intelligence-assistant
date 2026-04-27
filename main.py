"""
main.py
-------
Command-line interface for the Battery Time-Series Intelligence Assistant.

Usage:
    python main.py
    python main.py --dataset /path/to/custom.csv
"""

import sys
import argparse
from loader import load_battery_data
from preprocess import clean_data, detect_current_events
from features import compute_features, get_top_events
from plot import plot_voltage_current, plot_event_window, plot_temperature
import qa

DATASET = "/Users/arpitkhandelwal/Downloads/Research Project/Dataset Nasa.csv"

BANNER = """
╔══════════════════════════════════════════════════════╗
║   Battery Time-Series Intelligence Assistant  v1.0  ║
║   Type a question or command. 'help' to list all.  ║
╚══════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Available queries (natural language):
  voltage drop       — find the cell with the largest voltage drop
  temperature        — find highest thermal variation
  degradation        — health trend across all cells
  summary            — full diagnostic report
  events             — number of detected current events
  top events         — top 5 events ranked by di/dt intensity
  stats              — global dataset statistics

Plot commands:
  plot               — voltage + current overview
  plot temperature   — thermal profile
  plot event <N>     — zoom into event number N  (e.g. 'plot event 3')

Other:
  help               — show this menu
  exit / quit        — leave the assistant
"""


def build_pipeline(dataset_path: str):
    """Load, clean, and detect events. Return the prepared DataFrame."""
    print(f"\n[*] Loading: {dataset_path}")
    df = load_battery_data(dataset_path)
    df = clean_data(df)
    df = detect_current_events(df)
    print(f"[*] Ready — {len(df):,} rows, {df['cell_id'].nunique()} cell(s), "
          f"{int(df['event_id'].dropna().nunique())} events detected.\n")
    return df


def handle_plot_command(cmd: str, df):
    """Parse and dispatch plot commands."""
    cmd = cmd.strip().lower()
    if 'temperature' in cmd or 'temp' in cmd:
        plot_temperature(df)
    elif 'event' in cmd:
        # Try to extract event number from command, e.g. "plot event 3"
        match = __import__('re').search(r'\d+', cmd)
        if match:
            eid = float(match.group())
            plot_event_window(df, eid)
        else:
            print("[*] Specify an event number, e.g.  'plot event 3'")
    else:
        plot_voltage_current(df)


def run_cli(df):
    """Interactive REPL loop."""
    print(BANNER)
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[*] Exiting. Goodbye.")
            break

        if not user_input:
            continue

        low = user_input.lower()

        if low in ('exit', 'quit', 'q'):
            print("[*] Exiting. Goodbye.")
            break

        elif low in ('help', '?'):
            print(HELP_TEXT)

        elif low.startswith('plot'):
            handle_plot_command(low, df)

        else:
            response = qa.answer(user_input, df)
            # Print markdown-stripped version for terminal readability
            print("\nAssistant:\n" + _strip_markdown(response) + "\n")


def _strip_markdown(text: str) -> str:
    """Very lightweight markdown cleaner for terminal output."""
    import re
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)   # bold
    text = re.sub(r'`(.+?)`', r'\1', text)           # inline code
    text = re.sub(r'^#+\s+', '', text, flags=re.M)   # headings
    return text


def main():
    parser = argparse.ArgumentParser(description="Battery Intelligence Assistant — CLI")
    parser.add_argument('--dataset', default=DATASET,
                        help='Path to battery CSV file or directory.')
    args = parser.parse_args()

    try:
        df = build_pipeline(args.dataset)
        run_cli(df)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
