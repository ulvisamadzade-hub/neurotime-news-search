import os
import asyncio
import logging
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

from search import load_index, search
from keywords import load_metadata, extract_keywords

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_RESULTS = 5
MAX_SNIPPET = 180


def fmt_article(i: int, article: dict) -> str:
    date = article["published_at"][:10] if article["published_at"] else "?"
    score = article["relevance_score"]
    title = article["title"] or "Başlıq yoxdur"
    source = article["source"] or "?"
    snippet = (article["snippet"] or "")[:MAX_SNIPPET]
    url = article["url"] or ""
    return (
        f"*{i}. {title}*\n"
        f"📅 {date}  |  🌐 {source}  |  📊 {score:.2f}\n"
        f"{snippet}…\n"
        f"[Ətraflı oxu]({url})"
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Neurotime Xəbər Axtarış Assistanı\n\n"
        "Sadəcə istədiyiniz mövzunu yazın, məsələn:\n"
        "• _AccessBank haqqında xəbərlər tap_\n"
        "• _May 18-21 arasında gömrük xəbərləri_\n"
        "• _SOCAR-la bağlı mənfi xəbərlər_\n\n"
        "📌 Əmrlər:\n"
        "/keywords — ən çox rast gəlinən açar sözlər\n"
        "/help — kömək",
        parse_mode="Markdown",
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔍 *İstifadə qaydası*\n\n"
        "Hər hansı bir mövzu, şirkət, şəxs və ya tarix daxil edin.\n"
        "Tarix nümunələri:\n"
        "• _may 20-də_\n"
        "• _may 18 ilə 21 arasında_\n"
        "• _may 19-dan sonra_\n\n"
        "/keywords — qlobal açar söz siyahısı",
        parse_mode="Markdown",
    )


async def keywords_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Açar sözlər hesablanır...")
    kw = extract_keywords(top_n=15)
    lines = "\n".join(f"• *{term}*: {count}" for term, count in kw)
    await update.message.reply_text(
        f"🔑 *Ən çox rast gəlinən açar sözlər:*\n\n{lines}",
        parse_mode="Markdown",
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = (update.message.text or "").strip()
    if not query:
        return

    await update.message.reply_text("🔍 Axtarılır...")

    try:
        result = await asyncio.to_thread(search, query, top_k=MAX_RESULTS)
    except Exception as e:
        logger.error("Search error: %s", e)
        await update.message.reply_text("❌ Axtarış zamanı xəta baş verdi. Yenidən cəhd edin.")
        return

    articles = result.get("results", [])
    total = result.get("total_in_range", 0)
    params = result.get("params", {})

    date_info = ""
    if params.get("date_from") or params.get("date_to"):
        df = params.get("date_from", "başlanğıc")
        dt = params.get("date_to", "son")
        date_info = f"  |  📅 {df} → {dt}"

    if not articles:
        await update.message.reply_text(
            f"😔 Nəticə tapılmadı.\nMövzu: _{params.get('topic', query)}_{date_info}",
            parse_mode="Markdown",
        )
        return

    header = (
        f"📰 *{len(articles)} nəticə* (ümumi filtrə uyğun: {total})\n"
        f"🎯 Mövzu: _{params.get('topic', query)}_{date_info}\n"
        "─────────────────────"
    )
    await update.message.reply_text(header, parse_mode="Markdown")

    for i, article in enumerate(articles, 1):
        try:
            await update.message.reply_text(
                fmt_article(i, article),
                parse_mode="Markdown",
                disable_web_page_preview=True,
            )
        except Exception:
            # Fallback without markdown if formatting fails
            await update.message.reply_text(
                f"{i}. {article.get('title', '')}\n{article.get('url', '')}",
                disable_web_page_preview=True,
            )

    # Offer to show top keywords for these results
    keyboard = InlineKeyboardMarkup(
        [[InlineKeyboardButton("🔑 Bu nəticələrin açar sözləri", callback_data=f"kw::{query}")]]
    )
    await update.message.reply_text("Əlavə seçim:", reply_markup=keyboard)


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query_obj = update.callback_query
    await query_obj.answer()
    data = query_obj.data or ""

    if data.startswith("kw::"):
        search_query = data[4:]
        result = await asyncio.to_thread(search, search_query, top_k=50)
        snippets = [f"{r['title']} {r['snippet']}" for r in result["results"]]
        kw = extract_keywords(articles=snippets if snippets else None, top_n=15)
        lines = "\n".join(f"• *{term}*: {count}" for term, count in kw)
        await query_obj.message.reply_text(
            f"🔑 *Bu axtarış üçün açar sözlər:*\n\n{lines}",
            parse_mode="Markdown",
        )


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or token == "your_telegram_bot_token_here":
        raise ValueError("TELEGRAM_BOT_TOKEN is not set in .env")

    print("[bot] Loading search index...")
    load_index()
    load_metadata()
    print("[bot] Ready. Starting polling...")

    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("keywords", keywords_cmd))
    application.add_handler(CallbackQueryHandler(callback_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
