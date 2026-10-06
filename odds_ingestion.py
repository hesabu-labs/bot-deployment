# NOTE: retired from the active pipeline. bot_engine.py now runs a pure
# qualitative flow (fixtures + grounded LLM audit, no EV/accumulator math).
# This file is kept as a reference in case you want the quantitative screen
# back later - it still works standalone, just isn't called by the bot.

import os
import itertools
import pandas as pd
import requests
from google import genai

# --- Configuration ---
# Secrets now come from environment variables. Never hardcode keys in source.
# Create a .env file (see .env.example) and load it with python-dotenv, or
# export these in your shell / process manager before running this script.
ODDS_API_KEY = os.environ["ODDS_API_KEY"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_ALERT_CHAT_ID = os.environ.get("TELEGRAM_ALERT_CHAT_ID")
API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY")

# Standardized on the current google-genai SDK (was mixed with the legacy
# google.generativeai package across files - that package is deprecated).
client = genai.Client(api_key=GOOGLE_API_KEY)
GEMINI_MODEL = "gemini-3.8-flash"

SPORT_KEYS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league"
]

# Pinnacle serves as the global sharp benchmark; the rest proxy Kenyan market pricing
SHARP_BOOKMAKER = "pinnacle"
KENYAN_PROXIES = ["bet365", "1xbet", "betway", "unibet", "sportingbet"]
BOOKMAKERS = f"{SHARP_BOOKMAKER},{','.join(KENYAN_PROXIES)}"

# Standard balanced markets: 1X2, Over/Under (BTTS removed due to API tier restrictions)
MARKETS = "h2h,totals"
ODDS_FORMAT = "decimal"


def fetch_odds():
    """Fetches upcoming fixtures across target leagues from the EU/UK endpoints."""
    all_matches = []

    for sport in SPORT_KEYS:
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "eu,uk",
            "bookmakers": BOOKMAKERS,
            "markets": MARKETS,
            "oddsFormat": ODDS_FORMAT,
        }

        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                all_matches.extend(data)
                print(f"Successfully fetched {len(data)} fixtures for {sport}")
            else:
                print(f"Failed to fetch {sport}: {response.text}")
        except Exception as e:
            print(f"Error fetching {sport}: {e}")

    return all_matches


def calculate_all_markets_ev(
    fixtures_data, sharp_bookmaker=SHARP_BOOKMAKER, soft_books=KENYAN_PROXIES
):
    """
    Strips the vig from Pinnacle to get true probabilities, scans local proxies
    for the highest available decimal odds, and calculates Expected Value.
    """
    master_data = []

    for match in fixtures_data:
        match_name = f"{match['home_team']} vs {match['away_team']}"

        # 1. Isolate the Sharp Benchmark
        sharp_book = next(
            (b for b in match.get("bookmakers", []) if b["key"] == sharp_bookmaker),
            None,
        )
        if not sharp_book:
            continue

        # 2. Iterate through each sharp market (h2h, totals)
        for sharp_market in sharp_book.get("markets", []):
            market_key = sharp_market["key"]

            # Group outcomes by their point line to calculate vig independently
            point_groups = {}
            for outcome in sharp_market.get("outcomes", []):
                pt = outcome.get("point", "none")
                if pt not in point_groups:
                    point_groups[pt] = []
                point_groups[pt].append(outcome)

            for pt, outcomes in point_groups.items():
                # Calculate overround (the vig) for this specific outcome set
                implied_probs = {o["name"]: (1 / o["price"]) for o in outcomes}
                overround = sum(implied_probs.values())

                if overround == 0:
                    continue

                # Strip vig to calculate True Probability
                true_probs = {
                    name: (prob / overround) for name, prob in implied_probs.items()
                }

                # 3. Compare with local soft bookmakers
                for outcome_name, true_prob in true_probs.items():
                    best_local_price = 0
                    best_local_book = ""
                    display_name = f"{outcome_name} {pt if pt != 'none' else ''}".strip()

                    for book in match.get("bookmakers", []):
                        if book["key"] in soft_books:
                            soft_market = next(
                                (
                                    m
                                    for m in book.get("markets", [])
                                    if m["key"] == market_key
                                ),
                                None,
                            )
                            if soft_market:
                                soft_outcome = next(
                                    (
                                        o
                                        for o in soft_market.get("outcomes", [])
                                        if o["name"] == outcome_name
                                        and str(o.get("point", "none")) == str(pt)
                                    ),
                                    None,
                                )

                                if soft_outcome and soft_outcome["price"] > best_local_price:
                                    best_local_price = soft_outcome["price"]
                                    best_local_book = book["key"]

                    # Calculate Expected Value if a local price exists
                    if best_local_price > 0:
                        ev = (true_prob * best_local_price) - 1

                        master_data.append(
                            {
                                "match": match_name,
                                "market": f"{market_key} {pt if pt != 'none' else ''}".strip(),
                                "selection": display_name,
                                "true_prob": true_prob,
                                "local_odds": best_local_price,
                                "bookmaker": best_local_book,
                                "ev": ev,
                            }
                        )

    return pd.DataFrame(master_data)


def calculate_optimal_accumulators(positive_ev_df, min_odds=2.00, max_odds=3.00):
    """
    Combines positive EV single edges into 2-leg and 3-leg accumulators.
    Enforces unique match constraints to eliminate correlated outcomes.
    """
    single_bets = positive_ev_df.to_dict("records")
    valid_accumulators = []

    for leg_count in [2, 3]:
        for combo in itertools.combinations(single_bets, leg_count):

            # Enforce the Unique Match Constraint (no two legs from the same game)
            matches_in_combo = set([bet["match"] for bet in combo])
            if len(matches_in_combo) < leg_count:
                continue

            compound_odds = 1.0
            compound_true_prob = 1.0

            for bet in combo:
                compound_odds *= bet["local_odds"]
                compound_true_prob *= bet["true_prob"]

            # Filter for target odds range (2.00 - 3.00)
            if min_odds <= compound_odds <= max_odds:
                compound_ev = (compound_true_prob * compound_odds) - 1

                match_details = " + ".join(
                    [f"{b['match']} [{b['market']}: {b['selection']}]" for b in combo]
                )
                bookmakers = ", ".join(list(set([b["bookmaker"] for b in combo])))

                valid_accumulators.append(
                    {
                        "legs": leg_count,
                        "matches": match_details,
                        "bookmakers": bookmakers,
                        "odds": round(compound_odds, 2),
                        "compound_ev": round(compound_ev, 4),
                    }
                )

    return pd.DataFrame(valid_accumulators)


def send_telegram_alert(message):
    """Pushes the passing slip and risk report to your Telegram.
    Promoted to module level - previously nested inside unreachable code
    and never actually defined at runtime."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_ALERT_CHAT_ID:
        print("Telegram alert skipped: TELEGRAM_BOT_TOKEN / TELEGRAM_ALERT_CHAT_ID not configured.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_ALERT_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram Alert Failed: {e}")


def fetch_fundamental_metrics(fixture_id):
    """Pulls underlying fundamental data (xG, shots, possession) via API-Football.
    Promoted to module level - previously nested inside unreachable code.

    NOTE: this expects an API-Football fixture ID. The Odds API (used in
    fetch_odds) does NOT return that ID - the two providers use different
    fixture identifiers for the same match. You'll need a team-name/date
    matching step to resolve one to the other before this is usable; until
    then, callers should fall back to a baseline stats string.
    """
    if not API_FOOTBALL_KEY:
        return "Fundamental data unavailable (API_FOOTBALL_KEY not configured)."

    url = "https://v3.football.api-sports.io/fixtures/statistics"
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"fixture": fixture_id}

    try:
        response = requests.get(url, headers=headers, params=params)
        data = response.json()

        stats_text = ""
        if data.get('response'):
            for team_stats in data['response']:
                team_name = team_stats['team']['name']
                stats_text += f"\n--- {team_name} Underlying Metrics ---\n"
                for stat in team_stats['statistics']:
                    # Isolating key fundamental metrics
                    if stat['type'] in ["expected_goals", "Shots on Goal", "Ball Possession"]:
                        stats_text += f"{stat['type']}: {stat['value']}\n"
        return stats_text
    except Exception as e:
        print(f"Failed to fetch fundamental metrics: {e}")
        return "Fundamental data unavailable."


def run_qualitative_agent(slip, fundamental_stats="Data pending API integration."):
    """Runs the LLM risk audit on a single accumulator slip.

    Previously this function had an unreachable second try/except block
    after an unconditional return, which is where send_telegram_alert and
    fetch_fundamental_metrics used to live - meaning they never actually
    got defined and the script would crash with a NameError the first
    time a slip passed. Both helpers now live at module level above.
    """
    odds = slip['odds']
    matches = slip['matches']

    prompt = f"""
    You are an institutional sports betting risk analyst.
    I have identified a quantitative edge on the following accumulator slip at odds of {odds}:

    {matches}

    Below is the raw underlying fundamental performance data for the teams involved:
    {fundamental_stats}

    Your mandate is absolute capital preservation for a strict compounding ladder challenge where a single loss results in total portfolio ruin. Conduct a rigorous, unforgiving qualitative risk assessment evaluating the following friction points:

    1. Home and Away Form versus Current League Position.
    2. Historical Performance and Head-to-Head psychological advantages.
    3. Tactical and Stylistic Mismatches between the specific managers and systems.
    4. Motivation Asymmetry, contrasting relegation desperation against mid-table complacency.
    5. Key Injuries, Suspensions, and Mid-Week European Fixture Fatigue.
    6. Referee Assignments and Disciplinary Volatility.

    Synthesize these variables into a highly critical risk report. You must format your response as dense, scannable analytical prose. You are strictly forbidden from using bullet points or dashes anywhere in your output. Rely entirely on structured paragraphs to evaluate how these real-world variables either support or destroy the mathematical edge.

    If the selected teams face severe qualitative disadvantages, stylistic mismatches, or negative situational motivation, you must kill the trade.

    At the absolute end of your response, output a final verdict on a new line using strictly one of these two words:
    VERDICT: PASS
    VERDICT: REJECT
    """

    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return response.text
    except Exception as e:
        return f"Risk analysis failed: {e}\nVERDICT: REJECT"


# --- Execution Pipeline ---
if __name__ == "__main__":
    print("\n[1/4] Fetching market data...")
    fixtures = fetch_odds()

    print("\n[2/4] Processing probabilities and calculating Expected Value...")
    all_markets_df = calculate_all_markets_ev(fixtures)

    if all_markets_df.empty:
        print("\nNo fixtures with a usable Pinnacle benchmark were returned. Nothing to evaluate.")
    else:
        # Filter strictly for positive mathematical edge
        positive_ev_bets = all_markets_df[all_markets_df["ev"] > 0]

        if positive_ev_bets.empty:
            print("\nMarket is efficient. No positive EV bets found at this time.")
        else:
            print(f"\nFound {len(positive_ev_bets)} positive EV selections.")
            print(positive_ev_bets[['match', 'market', 'selection', 'local_odds', 'ev']].to_string(index=False))
            print("\n[3/4] Building optimal 2-leg and 3-leg accumulators...")
            optimal_slips = calculate_optimal_accumulators(
                positive_ev_bets, min_odds=1.80, max_odds=6.00
            )

            if optimal_slips.empty:
                print("\nNo combinations found even with relaxed odds. Check the single selections printed above.")
            else:
                print("\n--- Optimal Target Slips (EV Ranked) ---")
                pd.set_option("display.max_colwidth", None)
                print(
                    optimal_slips.sort_values(by="compound_ev", ascending=False)
                    .head(5)
                    .to_string(index=False)
                )

                # Define the efficient frontier: Top 10 slips ranked by Expected Value
                # (this used to run unconditionally even when optimal_slips was
                # empty/undefined - now properly nested under the checks above)
                frontier_slips = optimal_slips.sort_values(by="compound_ev", ascending=False).head(10)

                print(f"\n[4/4] Executing Qualitative Risk Agent on the Efficient Frontier ({len(frontier_slips)} Slips)...")

                for index, slip in frontier_slips.iterrows():
                    print("\n==================================================")
                    print(f"Evaluating Slip Profile: {slip['odds']} Odds | EV: {slip['compound_ev']:.4f}")
                    print(f"Matches: {slip['matches']}")
                    print("==================================================")

                    # Fetch stats if API-Football is linked, otherwise use baseline string
                    current_stats = "Evaluating baseline xG and tactical friction."
                    agent_analysis = run_qualitative_agent(slip, fundamental_stats=current_stats)

                    print("\n--- Contextual Agent Report ---")
                    print(agent_analysis)

                    if "VERDICT: PASS" in agent_analysis:
                        print("\n[+] Slip passed risk audit. Pushing alert to Telegram...")

                        alert_text = (
                            f"🚨 *LADDER FRONTIER: VALID SLIP* 🚨\n\n"
                            f"*Target Odds:* {slip['odds']} ({slip['bookmakers']})\n"
                            f"*Matches:*\n{slip['matches']}\n\n"
                            f"*Contextual Report:*\n{agent_analysis}"
                        )
                        send_telegram_alert(alert_text)
                    else:
                        print("\n[-] Slip rejected by qualitative risk agent. Moving to next option.")
