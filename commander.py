from telegram.ext import ApplicationBuilder, CommandHandler

from config import settings
from logging_config import setup_logger

logger = setup_logger("Commander")


async def start(update, context):
    await update.message.reply_text("🫡 Commander Online. Level 7.")


def run():
    app = ApplicationBuilder().token(settings.TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    logger.info("Commander Polling...")
    app.run_polling()


if __name__ == "__main__":
    run()
