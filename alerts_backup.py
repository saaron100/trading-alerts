import logging
import yfinance as yf
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# =====================
# Setup logging
# =====================
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================
# Replace with YOUR Telegram Bot Token
# =====================
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN_HERE"

# =====================
# Command Handlers
# =====================
def start(update, context):
    update.message.reply_text("👋 Hello! I’m your trading bot. Type 'help' to see commands.")

def help_command(update, context):
    commands = (
        "📌 Commands:\n"
        "- help → Show this list\n"
        "- test → Send a fake stock alert\n"
        "- portfolio → Show what you’re tracking\n"
        "- track (Ticker) → Start tracking (example: track TSLA)\n"
        "- stop (Ticker) → Stop tracking (example: stop TSLA)\n"
    )
    update.message.reply_text(commands)

def test_command(update, context):
    update.message.reply_text("🚨 TEST ALERT: (Ticker) CALL alert triggered ⭐⭐⭐⭐")

def portfolio_command(update, context):
    update.message.reply_text("📂 Your portfolio is empty (tracking will be added soon).")

def track_command(update, context):
    if len(context.args) == 0:
        update.message.reply_text("❌ Please give me a ticker. Example: track TSLA")
    else:
        ticker = context.args[0].upper()
        update.message.reply_text(f"✅ Now tracking {ticker}. (Alerts will come when conditions trigger.)")

def stop_command(update, context):
    if len(context.args) == 0:
        update.message.reply_text("❌ Please give me a ticker. Example: stop TSLA")
    else:
        ticker = context.args[0].upper()
        update.message.reply_text(f"🛑 Stopped tracking {ticker}.")

def unknown(update, context):
    update.message.reply_text("❓ I didn’t understand that. Type 'help' for commands.")

# =====================
# Main
# =====================
def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    # Commands
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(CommandHandler("test", test_command))
    dp.add_handler(CommandHandler("portfolio", portfolio_command))
    dp.add_handler(CommandHandler("track", track_command))
    dp.add_handler(CommandHandler("stop", stop_command))

    # Messages
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, unknown))

    # Start bot
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
