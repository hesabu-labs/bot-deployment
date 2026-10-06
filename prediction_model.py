"""
Lightweight Elo + independent-Poisson match predictor.

This is the same family of model outlets like Gracenote and various academic
papers use to forecast World Cup outcomes: a team-strength rating (Elo) feeds
expected-goal estimates, which feed a Poisson distribution over scorelines,
which is summed to get probabilities for whatever markets you care about.

Two honest simplifications versus a "real" version of this:
1. Elo ratings here start at a blank 1500 for every team and only move when
   YOU log a result via /result in the bot - there's no historical match
   backfill. It gets more useful the longer you use the bot, not on day one.
2. Goals are modeled as independent Poisson variables (home goals and away
   goals don't influence each other). Real models often use a bivariate
   Poisson or Dixon-Coles low-score correction, which does better on 0-0 and
   1-1 in particular. Left out here to keep this dependency-free and readable.
"""
import math
from itertools import product

HOME_ADVANTAGE_ELO = 60.0   # typical home-field Elo bump in club football
K_FACTOR = 20.0             # standard Elo update speed
LEAGUE_AVG_HOME_GOALS = 1.45
LEAGUE_AVG_AWAY_GOALS = 1.15
MAX_GOALS_MODELED = 8        # scoreline grid resolution (0..8 each side)


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _expected_score(elo_a: float, elo_b: float) -> float:
    """Standard Elo win-expectancy of A over B (0-1)."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def update_elo(home_elo: float, away_elo: float, home_goals: int, away_goals: int):
    """Returns (new_home_elo, new_away_elo) after a played result."""
    expected_home = _expected_score(home_elo + HOME_ADVANTAGE_ELO, away_elo)
    if home_goals > away_goals:
        actual_home = 1.0
    elif home_goals < away_goals:
        actual_home = 0.0
    else:
        actual_home = 0.5

    new_home_elo = home_elo + K_FACTOR * (actual_home - expected_home)
    new_away_elo = away_elo + K_FACTOR * ((1 - actual_home) - (1 - expected_home))
    return new_home_elo, new_away_elo


def predict_match(home_elo: float, away_elo: float) -> dict:
    """Elo -> expected goals -> Poisson scoreline grid -> market probabilities."""
    diff = (home_elo + HOME_ADVANTAGE_ELO) - away_elo

    # Elo diff -> a bounded multiplicative adjustment on the league-average
    # goal expectancy. The exponent (0.30) and clamp range are deliberately
    # conservative so a big rating gap nudges expected goals rather than
    # producing cartoonish scorelines.
    raw_ratio = 10 ** (diff / 400.0)
    adj = max(0.6, min(1.6, raw_ratio ** 0.30))

    expected_home_goals = LEAGUE_AVG_HOME_GOALS * adj
    expected_away_goals = LEAGUE_AVG_AWAY_GOALS / adj

    home_probs = [_poisson_pmf(k, expected_home_goals) for k in range(MAX_GOALS_MODELED + 1)]
    away_probs = [_poisson_pmf(k, expected_away_goals) for k in range(MAX_GOALS_MODELED + 1)]

    prob_home_win = prob_draw = prob_away_win = 0.0
    prob_btts_yes = 0.0
    prob_over_2_5 = 0.0
    best_score = (0, 0)
    best_score_prob = 0.0

    for h, a in product(range(MAX_GOALS_MODELED + 1), repeat=2):
        p = home_probs[h] * away_probs[a]
        if h > a:
            prob_home_win += p
        elif h < a:
            prob_away_win += p
        else:
            prob_draw += p
        if h > 0 and a > 0:
            prob_btts_yes += p
        if h + a >= 3:
            prob_over_2_5 += p
        if p > best_score_prob:
            best_score_prob = p
            best_score = (h, a)

    return {
        "home_elo": round(home_elo, 1),
        "away_elo": round(away_elo, 1),
        "expected_home_goals": round(expected_home_goals, 2),
        "expected_away_goals": round(expected_away_goals, 2),
        "prob_home_win": round(prob_home_win, 3),
        "prob_draw": round(prob_draw, 3),
        "prob_away_win": round(prob_away_win, 3),
        "prob_btts_yes": round(prob_btts_yes, 3),
        "prob_over_2_5": round(prob_over_2_5, 3),
        "most_likely_score": f"{best_score[0]}-{best_score[1]}",
        "most_likely_score_prob": round(best_score_prob, 3),
    }


def format_prediction_for_prompt(home_team: str, away_team: str, pred: dict) -> str:
    """Turns the numeric prediction into a plain-language block for the LLM prompt."""
    return f"""
    STATISTICAL BASELINE (Elo + Poisson goal model - a prior, not a verdict):
    {home_team} Elo: {pred['home_elo']} | {away_team} Elo: {pred['away_elo']}
    Expected goals: {home_team} {pred['expected_home_goals']} - {pred['expected_away_goals']} {away_team}
    Win/Draw/Win probabilities: {home_team} {pred['prob_home_win']*100:.1f}% / Draw {pred['prob_draw']*100:.1f}% / {away_team} {pred['prob_away_win']*100:.1f}%
    BTTS Yes probability: {pred['prob_btts_yes']*100:.1f}%
    Over 2.5 goals probability: {pred['prob_over_2_5']*100:.1f}%
    Most likely scoreline: {pred['most_likely_score']} ({pred['most_likely_score_prob']*100:.1f}% chance of this exact score)

    Note: this baseline is only as good as the Elo ratings behind it, which
    improve as more real results get logged. Treat it as a starting prior to
    confirm, adjust, or override using real current information - not as a
    finished answer.
    """
