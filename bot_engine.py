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
ODDS_API_KEY = os.environ["ODDS_API_KEY"]  
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
MAX_FIXTURES_SHOWN = 30  

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
        "/result <audit_id> win|loss [home-away score] - log a real outcome\n"
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
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 2:
        await update.message.reply_text("Usage: /result <audit_id> win|loss [home-away score, e.g. 2-1]")
        return

    try:
        audit_id = int(args[0])
    except ValueError:
        await update.message.reply_text("audit_id must be a number - check /history for IDs.")
        return

    outcome = args[1].lower()
    if outcome not in ("win", "loss"):
        await update.message.reply_text("Outcome must be 'win' or 'loss'.")
        return

    audit = storage.get_audit(audit_id)
    if not audit or audit["chat_id"] != chat_id:
        await update.message.reply_text("Audit ID not found.")
        return

    storage.set_audit_outcome(audit_id, outcome)
    msg = f"Logged audit #{audit_id} as {outcome}."

    if len(args) >= 3 and re.match(r"^\d+-\d+$", args[2]):
        hg, ag = (int(x) for x in args[2].split("-"))
        home_elo = storage.get_elo(audit["home_team"])
        away_elo = storage.get_elo(audit["away_team"])
        new_home, new_away = pm.update_elo(home_elo, away_elo, hg, ag)
        storage.set_elo(audit["home_team"], new_home)
        storage.set_elo(audit["away_team"], new_away)
        msg += (f" Elo updated from final score {hg}-{ag}: "
                f"{audit['home_team']} {home_elo:.0f}→{new_home:.0f}, "
                f"{audit['away_team']} {away_elo:.0f}→{new_away:.0f}.")
    else:
        msg += " (No score given, so Elo ratings weren't updated this time.)"

    await update.message.reply_text(msg)


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    s = storage.audit_stats(chat_id)
    if s["pass_total"] == 0 and s["reject_total"] == 0:
        await update.message.reply_text("No logged outcomes yet. Use /result <audit_id> win|loss after each bet settles.")
        return
    lines = ["📊 *Audit track record (self-reported)*"]
    if s["pass_total"]:
        rate = s["pass_win"] / s["pass_total"] * 100
        lines.append(f"PASS calls: {s['pass_total']}, hit rate {rate:.0f}%")
    if s["reject_total"]:
        lines.append(f"REJECT calls: {s['reject_total']}, would-have-won {s['reject_would_have_won']}")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def stake_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("Usage: /stake <amount>")
        return
    try:
        amount = float(context.args[0])
    except ValueError:
        await update.message.reply_text("Amount must be a number.")
        return
    existing = storage.get_open_stake(chat_id)
    if existing:
        await update.message.reply_text(
            f"You already have an open stake of {existing['stake']} from {existing['created_at'][:10]}. "
            f"Settle it first with /settle win <payout> or /settle loss."
        )
        return
    lid = storage.open_stake(chat_id, amount)
    await update.message.reply_text(f"Stake #{lid} of {amount} opened for this week.")


async def settle_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("Usage: /settle win <payout> | /settle loss")
        return
    existing = storage.get_open_stake(chat_id)
    if not existing:
        await update.message.reply_text("No open stake to settle. Start one with /stake <amount>.")
        return
    status = context.args[0].lower()
    if status not in ("win", "loss"):
        await update.message.reply_text("First argument must be 'win' or 'loss'.")
        return
    payout = 0.0
    if status == "win":
        if len(context.args) < 2:
            await update.message.reply_text("Usage: /settle win <payout amount>")
            return
        try:
            payout = float(context.args[1])
        except ValueError:
            await update.message.reply_text("Payout must be a number.")
            return
    storage.settle_stake(existing["id"], status, payout)
    extra = f" - payout {payout}" if status == "win" else ""
    await update.message.reply_text(f"Stake #{existing['id']} settled as {status}{extra}.")


async def ledger_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    rows = storage.list_ledger(chat_id)
    if not rows:
        await update.message.reply_text("No ledger entries yet. Start with /stake <amount>.")
        return
    lines = ["📒 *Ledger (most recent first)*"]
    for r in rows:
        if r["status"] == "open":
            lines.append(f"#{r['id']} {r['week']}: staked {r['stake']}, OPEN")
        else:
            lines.append(f"#{r['id']} {r['week']}: staked {r['stake']} → {r['status'].upper()} (payout {r['payout']})")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def quota_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    used, cap = storage.get_quota_usage(chat_id, GEMINI_WEEKLY_QUOTA)
    await update.message.reply_text(f"Gemini calls used this week: {used}/{cap}.")


# ---------------------------------------------------------------------------
# Callback (button) handler
# ---------------------------------------------------------------------------

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    chat_id = query.message.chat_id
    data = query.data
    await query.answer()

    if data.startswith("tog:"):
        fid = int(data.split(":")[1])
        sess = SESSIONS.get(chat_id)
        if not sess:
            await query.edit_message_text("Session expired - run /audit again.")
            return
        sess["selected"].symmetric_difference_update({fid})
        await query.edit_message_reply_markup(reply_markup=_fixture_keyboard(chat_id))

    elif data == "runaudit":
        sess = SESSIONS.get(chat_id)
        if not sess or not sess["selected"]:
            await query.edit_message_text("No matches selected. Run /audit again to pick some.")
            return
        chosen = [f for f in sess["fixtures"] if f["id"] in sess["selected"]]
        text = "Confirm audit on:\n" + "\n".join(f"• {f['match']}" for f in chosen)
        text += f"\n\nThis uses {len(chosen)} Gemini call(s) with search grounding (extra cost per call)."
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Confirm", callback_data="confirmaudit"),
            InlineKeyboardButton("✖ Cancel", callback_data="cancelaudit"),
        ]])
        await query.edit_message_text(text, reply_markup=kb)

    elif data == "confirmaudit":
        sess = SESSIONS.get(chat_id)
        if not sess:
            await query.edit_message_text("Session expired - run /audit again.")
            return
        chosen = [f for f in sess["fixtures"] if f["id"] in sess["selected"]]
        await query.edit_message_text(f"Running audit on {len(chosen)} match(es)...")
        await run_audits(chat_id, chosen, context.bot)

    elif data == "cancelaudit":
        SESSIONS.pop(chat_id, None)
        await query.edit_message_text("Audit cancelled.")

    elif data.startswith("full:"):
        audit_id = int(data.split(":")[1])
        audit = storage.get_audit(audit_id)
        if not audit:
            await query.answer("Not found", show_alert=True)
            return
        await send_chunked(context.bot, chat_id, f"📄 *Full analysis — {audit['match']}*\n\n{audit['full_analysis']}")

    elif data.startswith("mkt:"):
        m = data.split(":", 1)[1]
        enabled = set(storage.get_market_prefs(chat_id))
        if m in enabled and len(enabled) > 1:
            enabled.discard(m)
        elif m not in enabled:
            enabled.add(m)
        storage.set_market_prefs(chat_id, [x for x in ALL_MARKETS if x in enabled])
        await query.edit_message_reply_markup(reply_markup=_markets_keyboard(chat_id))

    elif data == "mktdone":
        await query.edit_message_text("Market preferences saved. Run /markets any time to change them.")


def main():
    storage.init_db()
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("audit", audit_command))
    app.add_handler(CommandHandler("markets", markets_command))
    app.add_handler(CommandHandler("favorite", favorite_command))
    app.add_handler(CommandHandler("unfavorite", unfavorite_command))
    app.add_handler(CommandHandler("favorites", favorites_command))
    app.add_handler(CommandHandler("history", history_command))
    app.add_handler(CommandHandler("result", result_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("stake", stake_command))
    app.add_handler(CommandHandler("settle", settle_command))
    app.add_handler(CommandHandler("ledger", ledger_command))
    app.add_handler(CommandHandler("quota", quota_command))
    app.add_handler(CallbackQueryHandler(button_callback))

    print("[+] Bot core initialized (pure qualitative, V2). Polling for commands...")
    app.run_polling()


if __name__ == "__main__":
    main()
