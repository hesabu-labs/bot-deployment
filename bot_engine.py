import os
import re
import logging
import asyncio
from datetime import datetime, timezone

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
)
from google import genai
from google.genai import types

import storage
import prediction_model as pm

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
ODDS_API_KEY = os.environ["ODDS_API_KEY"]  # only used for the free /events fixture list
GEMINI_WEEKLY_QUOTA = int(os.environ.get("GEMINI_WEEKLY_QUOTA", "60"))

client = genai.Client(api_key=GOOGLE_API_KEY)
GEMINI_MODEL = "gemini-3.8-flash"
GROUNDING_CONFIG = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())]
)

SPORT_KEYS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
]

ALL_MARKETS = ["1X2", "BTTS", "Goals", "Corners"]
MAX_FIXTURES_SHOWN = 30  # keeps the inline keyboard usable - Telegram gets unwieldy well before 100 rows

# Ephemeral in-memory session state for the audit selection flow.
# Not persisted - a bot restart mid-selection just means re-running /audit.
SESSIONS = {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _format_kickoff(commence_time_iso: str) -> str:
    try:
        dt = datetime.fromisoformat(commence_time_iso.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%a %d %b, %H:%M UTC")
    except Exception:
        return commence_time_iso


def fetch_upcoming_fixtures():
    """Real upcoming fixtures from The Odds API's /events endpoint - metadata
    only, no bookmaker odds, no per-request odds cost."""
    import requests

    all_fixtures = []
    next_id = 1

    for sport in SPORT_KEYS:
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/events"
        params = {"apiKey": ODDS_API_KEY}
        try:
            response = requests.get(url, params=params)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch events for {sport}: {response.text}")
                continue
            for event in response.json():
                all_fixtures.append({
                    "id": next_id,
                    "match": f"{event['home_team']} vs {event['away_team']}",
                    "home_team": event["home_team"],
                    "away_team": event["away_team"],
                    "kickoff": _format_kickoff(event.get("commence_time", "")),
                })
                next_id += 1
        except Exception as e:
            logger.warning(f"Error fetching {sport}: {e}")

    return all_fixtures


def _team_matches(fav: str, team_name: str) -> bool:
    fav, team_name = fav.lower(), team_name.lower()
    return fav in team_name or team_name in fav


# ---------------------------------------------------------------------------
# Inline keyboards
# ---------------------------------------------------------------------------

def _fixture_keyboard(chat_id):
    sess = SESSIONS[chat_id]
    fixtures = sess["fixtures"]
    selected = sess["selected"]

    rows = []
    row = []
    for f in fixtures:
        mark = "✅" if f["id"] in selected else "☐"
        star = "⭐" if f.get("is_favorite") else ""
        label = f"{mark}{star} {f['match']}"[:64]
        row.append(InlineKeyboardButton(label, callback_data=f"tog:{f['id']}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)

    rows.append([
        InlineKeyboardButton(f"▶ Run Audit ({len(selected)})", callback_data="runaudit"),
        InlineKeyboardButton("✖ Cancel", callback_data="cancelaudit"),
    ])
    return InlineKeyboardMarkup(rows)


def _markets_keyboard(chat_id):
    enabled = set(storage.get_market_prefs(chat_id))
    rows = []
    for m in ALL_MARKETS:
        mark = "✅" if m in enabled else "☐"
        rows.append([InlineKeyboardButton(f"{mark} {m}", callback_data=f"mkt:{m}")])
    rows.append([InlineKeyboardButton("Done", callback_data="mktdone")])
    return InlineKeyboardMarkup(rows)


# ---------------------------------------------------------------------------
# Prompt building + parsing
# ---------------------------------------------------------------------------

def build_match_prompt(match: dict, prediction_block: str, markets_pref: list) -> str:
    markets_list = "\n".join(f"    {m}" for m in markets_pref)
    return f"""
    You are an institutional sports betting risk analyst.

    {prediction_block}

    Before writing your qualitative analysis, use Google Search to find current,
    dated information about this specific fixture: {match['match']}, kicking off
    {match['kickoff']}. Look specifically for: current league form and table
    position for both sides, head-to-head history, confirmed injuries or
    suspensions, mid-week fixture fatigue, and the confirmed or likely referee
    if available. If you cannot find reliable current information on a point,
    say so explicitly rather than inventing it.

    Weigh the statistical baseline above against what you find. The baseline is
    a simple Elo/Poisson prior and knows nothing about injuries, suspensions,
    or news - your job is to confirm it, adjust it, or override it, and say
    explicitly which you are doing and why.

    Your mandate is absolute capital preservation for a strict compounding
    ladder challenge where a single loss results in total portfolio ruin.

    Respond in EXACTLY this structure, with each header on its own line and
    nothing before SUMMARY or after VERDICT:

    SUMMARY: one or two sentences - your overall qualitative read versus the statistical baseline.
    ANALYSIS: dense, scannable analytical prose, no bullet points or dashes, covering the friction points you found via search and how they support or undermine the statistical baseline.
    MARKETS: an explicit recommendation for each of the following markets, one per line -
    {markets_list}
    VERDICT: strictly one word, PASS or REJECT, alone on the final line.
    """


_REPORT_TAGS = ["SUMMARY", "ANALYSIS", "MARKETS", "VERDICT"]


def parse_report(raw_text: str) -> dict:
    def extract(tag, text):
        other_tags = "|".join(t for t in _REPORT_TAGS if t != tag)
        pattern = rf"{tag}:\s*(.*?)(?=\n(?:{other_tags}):|\Z)"
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).strip() if m else ""

    summary = extract("SUMMARY", raw_text)
    markets = extract("MARKETS", raw_text)
    analysis = extract("ANALYSIS", raw_text)
    verdict_match = re.search(r"VERDICT:\s*(PASS|REJECT)", raw_text)
    verdict = verdict_match.group(1) if verdict_match else "REJECT"

    if not summary and not markets and not analysis:
        analysis = raw_text
        summary = (raw_text[:280] + "...") if len(raw_text) > 280 else raw_text
        markets = ""

    return {"summary": summary, "markets": markets, "analysis": analysis, "verdict": verdict}


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------

async def run_grounded_audit(prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt, config=GROUNDING_CONFIG
        )
        return response.text
    except Exception as e:
        error_msg = str(e)
        if any(code in error_msg for code in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
            logger.warning(f"API Congestion/Rate Limit hit. Backing off for 5s... ({error_msg})")
            await asyncio.sleep(5)
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL, contents=prompt, config=GROUNDING_CONFIG
                )
                return response.text
            except Exception as retry_e:
                return f"Endpoint still congested after retry: {retry_e}\nVERDICT: REJECT"
        return f"Audit failed: {e}\nVERDICT: REJECT"


async def send_chunked(bot, chat_id, text):
    for i in range(0, len(text), 4000):
        chunk = text[i:i + 4000]
        try:
            await bot.send_message(chat_id, chunk, parse_mode="Markdown")
        except Exception:
            await bot.send_message(chat_id, chunk)


async def run_audits(chat_id, chosen_matches, bot):
    markets_pref = storage.get_market_prefs(chat_id)

    for i, match in enumerate(chosen_matches):
        if i > 0:
            await asyncio.sleep(4)
            
        allowed, used, cap = storage.check_and_increment_quota(chat_id, GEMINI_WEEKLY_QUOTA)
        if not allowed:
            await bot.send_message(
                chat_id,
                f"⛔ Weekly Gemini quota reached ({used}/{cap}). Skipping remaining matches. "
                f"Adjust GEMINI_WEEKLY_QUOTA in your env if you want more headroom."
            )
            break

        home_elo = storage.get_elo(match["home_team"])
        away_elo = storage.get_elo(match["away_team"])
        prediction = pm.predict_match(home_elo, away_elo)
        prediction_block = pm.format_prediction_for_prompt(match["home_team"], match["away_team"], prediction)

        await bot.send_message(chat_id, f"🔎 Researching: {match['match']}...")
        prompt = build_match_prompt(match, prediction_block, markets_pref)
        raw_text = await run_grounded_audit(prompt)
        parsed = parse_report(raw_text)

        audit_id = storage.record_audit(
            chat_id=chat_id, match=match["match"], home_team=match["home_team"], away_team=match["away_team"],
            verdict=parsed["verdict"], summary=parsed["summary"], markets=parsed["markets"],
            full_analysis=parsed["analysis"], prediction=prediction,
        )

        badge = "✅ PASS" if parsed["verdict"] == "PASS" else "❌ REJECT"
        card = (
            f"{badge} — *{match['match']}*\n\n"
            f"_Model prior:_ {match['home_team']} {prediction['prob_home_win']*100:.0f}% / "
            f"Draw {prediction['prob_draw']*100:.0f}% / {match['away_team']} {prediction['prob_away_win']*100:.0f}%"
            f"  (likely {prediction['most_likely_score']})\n\n"
            f"{parsed['summary']}\n\n"
            f"{parsed['markets']}\n\n"
            f"_Audit #{audit_id} - log the real result later:_ `/result {audit_id} win|loss 2-1`"
        )
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("📄 Full analysis", callback_data=f"full:{audit_id}")]])
        try:
            await bot.send_message(chat_id, card, parse_mode="Markdown", reply_markup=kb)
        except Exception:
            await bot.send_message(chat_id, card, reply_markup=kb)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "🚨 *LADDER CHALLENGE COMMAND DESK* 🚨\n\n"
        "Pure qualitative mode - no market pricing, no accumulators, one grounded "
        "audit per match with an Elo/Poisson statistical prior underneath it.\n\n"
        "Type /help to see everything."
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "*Commands*\n"
        "/audit - pull this week's fixtures, tap to select, then Run Audit\n"
        "/markets - choose which markets get a recommendation\n"
        "/favorite <team> - pin a team so their fixtures surface first\n"
        "/unfavorite <team> - remove a pin\n"
        "/favorites - list pinned teams\n"
        "/history - your last 10 audits\n"
        "/result <audit\_id> win|loss [home-away score] - log a real outcome\n"
        "/stats - PASS/REJECT hit rate from logged outcomes\n"
        "/stake <amount> - open this week's stake\n"
        "/settle win <payout> | /settle loss - close the open stake\n"
        "/ledger - recent stakes and results\n"
        "/quota - Gemini calls used this week"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def audit_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text("Pulling this week's fixtures...")
    fixtures = fetch_upcoming_fixtures()

    if not fixtures:
        await update.message.reply_text("No fixtures came back. Check ODDS_API_KEY / quota and try again.")
        return

    favorites = storage.list_favorites(chat_id)
    for f in fixtures:
        f["is_favorite"] = any(_team_matches(fav, f["home_team"]) or _team_matches(fav, f["away_team"])
                                for fav in favorites)
    fixtures.sort(key=lambda f: (not f["is_favorite"], f["id"]))
    fixtures = fixtures[:MAX_FIXTURES_SHOWN]

    SESSIONS[chat_id] = {"fixtures": fixtures, "selected": set()}

    await update.message.reply_text(
        f"🔍 *SELECT GAMES FOR QUALITATIVE AUDIT* 🔍 (showing {len(fixtures)})\n"
        f"Tap to toggle, then Run Audit.",
        parse_mode="Markdown",
        reply_markup=_fixture_keyboard(chat_id),
    )


async def markets_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text(
        "Toggle which markets the audit recommends on:",
        reply_markup=_markets_keyboard(chat_id),
    )


async def favorite_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("Usage: /favorite <team name>")
        return
    team = " ".join(context.args)
    storage.add_favorite(chat_id, team)
    await update.message.reply_text(f"⭐ Added {team}. Their fixtures surface first in /audit.")


async def unfavorite_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("Usage: /unfavorite <team name>")
        return
    team = " ".join(context.args)
    storage.remove_favorite(chat_id, team)
    await update.message.reply_text(f"Removed {team} from favorites.")


async def favorites_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    favs = storage.list_favorites(chat_id)
    if not favs:
        await update.message.reply_text("No favorites yet. Add one with /favorite <team name>.")
        return
    await update.message.reply_text("⭐ " + "\n⭐ ".join(favs))


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    rows = storage.list_recent_audits(chat_id, limit=10)
    if not rows:
        await update.message.reply_text("No audits yet. Run /audit to get started.")
        return
    lines = ["🗂 *Recent audits*"]
    for r in rows:
        badge = "✅" if r["verdict"] == "PASS" else "❌"
        outcome = f" → {r['outcome']}" if r["outcome"] else ""
        lines.append(f"#{r['id']} {badge} {r['match']}{outcome}")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def result_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
