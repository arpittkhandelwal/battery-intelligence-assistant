"""
Commands:
    /start           — welcome + help
    /stats           — global dataset statistics
    /events          — list all detected current-change events
    /summary         — full diagnostic report
    /degradation     — voltage-trend health table
    /voltage_drop    — cell with largest voltage drop
    /temperature     — cell with highest thermal variation
    /plot            — voltage + current chart (image)
    /plot_event <N>  — zoom into event number N  (e.g. /plot_event 3)
    Free text        — routed through the rule-based QA engine
"""

import os
import io
import logging
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no GUI required
import matplotlib.pyplot as plt

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from loader import load_battery_data
from preprocess import clean_data, detect_current_events
from features import compute_features, get_top_events
import qa
from plot import plot_voltage_current, plot_event_window, plot_temperature

# ---------------------------------------------------------------------------
# Configuration — replace with your token from @BotFather
# ---------------------------------------------------------------------------
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8726595592:AAE_GKbkM5IfHsXY_7tEQDA2NJzO94hssME")
DATASET_PATH = "Dataset Nasa.csv"

# ---------------------------------------------------------------------------
# Boot-time pipeline (runs once at startup, shared across all handlers)
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
log = logging.getLogger(__name__)

def _load_pipeline(path: str):
    """Load and process the dataset. Called once at startup."""
    log.info(f"Loading dataset: {path}")
    df = load_battery_data(path)
    df = clean_data(df)
    df = detect_current_events(df)
    log.info(f"Pipeline ready — {len(df):,} rows, "
             f"{int(df['event_id'].dropna().nunique())} events detected.")
    return df

DF = _load_pipeline(DATASET_PATH)

HELP_TEXT = (
    "*Battery Intelligence Assistant*\n\n"
    "*Commands:*\n"
    "/stats — global dataset statistics\n"
    "/events — all detected current-change events\n"
    "/summary — full diagnostic report\n"
    "/degradation — voltage health trend\n"
    "/voltage\\_drop — largest voltage drop\n"
    "/temperature — highest thermal variation\n"
    "/plot — voltage + current chart\n"
    "/plot\\_event N — zoom into event N  *(e.g. /plot\\_event 3)*\n\n"
    "*Or just type a question:*\n"
    "  _Which cell has the largest voltage drop?_\n"
    "  _Give me the top events_\n"
    "  _Show the degradation trend_"
)

# ---------------------------------------------------------------------------
# Helper: send a matplotlib figure as a Telegram photo
# ---------------------------------------------------------------------------
def _fig_to_bytes(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='#1a1a2e')
    buf.seek(0)
    plt.close(fig)
    return buf


def _make_voltage_plot(df, title="Battery Telemetry") -> io.BytesIO:
    """Build a dark-themed voltage + current dual-axis chart."""
    fig, ax1 = plt.subplots(figsize=(12, 5), facecolor='#1a1a2e')
    ax1.set_facecolor('#1a1a2e')

    ax1.set_xlabel("Time (s)", color='#aaaaaa')
    ax1.set_ylabel("Voltage (V)", color="#4fc3f7")
    ax1.plot(df['time'], df['voltage'], color="#4fc3f7", lw=1.4, label="Voltage")
    ax1.tick_params(axis='both', colors='#aaaaaa')
    ax1.spines['bottom'].set_color('#444')
    ax1.spines['left'].set_color('#444')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2 = ax1.twinx()
    ax2.set_facecolor('#1a1a2e')
    ax2.set_ylabel("Current (A)", color="#ef5350")
    ax2.plot(df['time'], df['current'], color="#ef5350", lw=0.8, alpha=0.5, label="Current")
    ax2.tick_params(axis='y', colors='#aaaaaa')
    ax2.spines['right'].set_color('#444')

    if 'is_event' in df.columns:
        ev = df[df['is_event']]
        ax1.scatter(ev['time'], ev['voltage'], color='orange',
                    s=20, zorder=5, label='Events')

    ax1.set_title(title, color='white', pad=10)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def _make_event_plot(df, event_id: float) -> io.BytesIO:
    """Build a zoomed event window chart."""
    from preprocess import extract_event_window
    window = extract_event_window(df, event_id)
    if window.empty:
        return None
    return _make_voltage_plot(window, title=f"Event {int(event_id)} — Zoom")

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT, parse_mode='Markdown')


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT, parse_mode='Markdown')


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response = qa.answer("global stats", DF)
    await update.message.reply_text(response, parse_mode='Markdown')


async def cmd_events(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response = qa.answer("give me the top events", DF)
    await update.message.reply_text(response, parse_mode='Markdown')


async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response = qa.answer("summary report", DF)
    await update.message.reply_text(response, parse_mode='Markdown')


async def cmd_degradation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response = qa.answer("degradation trend", DF)
    await update.message.reply_text(response, parse_mode='Markdown')


async def cmd_voltage_drop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response = qa.answer("voltage drop", DF)
    await update.message.reply_text(response, parse_mode='Markdown')


async def cmd_temperature(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response = qa.answer("temperature", DF)
    await update.message.reply_text(response, parse_mode='Markdown')


async def cmd_plot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Generating chart…")
    buf = _make_voltage_plot(DF)
    await update.message.reply_photo(photo=buf, caption="Voltage & Current — Full Timeline")


async def cmd_plot_event(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Usage: /plot_event <N>"""
    args = context.args
    if not args or not args[0].isdigit():
        await update.message.reply_text(
            "Usage: `/plot_event N`  where N is an event number.\n"
            "Use /events to see available event IDs.",
            parse_mode='Markdown'
        )
        return

    event_id = float(args[0])
    valid_ids = sorted(DF['event_id'].dropna().unique().astype(int))

    if int(event_id) not in valid_ids:
        await update.message.reply_text(
            f"Event {int(event_id)} not found.\nAvailable events: {valid_ids}"
        )
        return

    await update.message.reply_text(f"⏳ Generating zoom for event {int(event_id)}…")
    buf = _make_event_plot(DF, event_id)
    if buf is None:
        await update.message.reply_text("No data found for that event window.")
        return
    await update.message.reply_photo(
        photo=buf,
        caption=f"Event {int(event_id)} — Voltage & Current Zoom"
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Route free-text messages through the QA engine."""
    question = update.message.text.strip()
    log.info(f"User: {question}")
    
    intent = qa._score_intent(question)
    
    # Intercept plot commands so we can send actual images
    if intent == 'plot':
        q_lower = question.lower()
        # Check if they want to plot a specific/strongest event
        if any(word in q_lower for word in ['strongest', 'top', 'biggest', 'worst', 'event']):
            top = get_top_events(DF, n=1)
            if not top.empty:
                event_id = top.iloc[0]['event_id']
                await update.message.reply_text(f"⏳ Generating zoom for strongest event ({int(event_id)})…")
                buf = _make_event_plot(DF, event_id)
                if buf:
                    await update.message.reply_photo(
                        photo=buf,
                        caption=f"Event {int(event_id)} (Strongest) — Voltage & Current Zoom"
                    )
                else:
                    await update.message.reply_text("Could not extract event window.")
                return
                
        # Default full-timeline plot
        await update.message.reply_text("⏳ Generating chart…")
        buf = _make_voltage_plot(DF)
        await update.message.reply_photo(photo=buf, caption="Voltage & Current — Full Timeline")
        return

    # All other intents return standard text
    response = qa.answer(question, DF)
    await update.message.reply_text(response, parse_mode='Markdown')

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

from telegram.request import HTTPXRequest

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        html = """
        <html>
            <head><title>Battery Assistant Status</title></head>
            <body style="font-family: sans-serif; text-align: center; padding-top: 50px; background: #1a1a2e; color: white;">
                <h1>Battery Intelligence Assistant</h1>
                <p style="color: #4fc3f7; font-size: 1.2em;">Telegram API is Active and Connected</p>
                <div style="margin: 20px; padding: 20px; border: 1px solid #444; border-radius: 10px; display: inline-block;">
                    <p>Status: <b>Active</b></p>
                    <p>Events Processed: <b>16</b></p>
                </div>
                <p>Chat with the bot on Telegram to analyze telemetry.</p>
            </body>
        </html>
        """
        self.wfile.write(html.encode())
    def log_message(self, format, *args): return # Silence logs

def run_health_server():
    server = HTTPServer(('0.0.0.0', 7860), HealthCheckHandler)
    server.serve_forever()

def main():
    # Start health check server in background to satisfy Hugging Face
    threading.Thread(target=run_health_server, daemon=True).start()
    log.info("Health check server started on port 7860")

    if BOT_TOKEN == "YOUR_TOKEN_HERE":
        print("\n[ERROR] Please set your Telegram bot token.")
        print("  Option 1 — Edit bot.py and replace 'YOUR_TOKEN_HERE'")
        print("  Option 2 — Run:  TELEGRAM_BOT_TOKEN=<token> python bot.py\n")
        return

    # Use explicit HTTPXRequest for robust cloud connections
    t_request = HTTPXRequest(connect_timeout=60, read_timeout=60)
    app = ApplicationBuilder().token(BOT_TOKEN).request(t_request).build()

    app.add_handler(CommandHandler("start",       cmd_start))
    app.add_handler(CommandHandler("help",        cmd_help))
    app.add_handler(CommandHandler("stats",       cmd_stats))
    app.add_handler(CommandHandler("events",      cmd_events))
    app.add_handler(CommandHandler("summary",     cmd_summary))
    app.add_handler(CommandHandler("degradation", cmd_degradation))
    app.add_handler(CommandHandler("voltage_drop",cmd_voltage_drop))
    app.add_handler(CommandHandler("temperature", cmd_temperature))
    app.add_handler(CommandHandler("plot",        cmd_plot))
    app.add_handler(CommandHandler("plot_event",  cmd_plot_event))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    log.info("Bot is running. Press Ctrl+C to stop.")
    # Add bootstrap retries for unstable cloud network startups
    app.run_polling(bootstrap_retries=10)


if __name__ == "__main__":
    main()
