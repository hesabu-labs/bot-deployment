"""
SQLite persistence layer for the ladder bot.
Stdlib only (sqlite3) - no new external dependency.
One file, one DB: ladder_bot.db, created next to wherever this runs.
"""
import os
import sqlite3
import json
from datetime import datetime, timezone
from contextlib import contextmanager

# Anchor the DB path to the directory where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "ladder_bot.db")
DEFAULT_MARKETS = ["1X2", "BTTS", "Goals", "Corners"]


def init_db():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS elo_ratings (
            team TEXT PRIMARY KEY,
            rating REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            match TEXT NOT NULL,
            home_team TEXT,
            away_team TEXT,
            verdict TEXT,
            summary TEXT,
            markets TEXT,
            full_analysis TEXT,
            prediction_json TEXT,
            created_at TEXT NOT NULL,
            outcome TEXT
        );

        CREATE TABLE IF NOT EXISTS ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            week TEXT NOT NULL,
            stake REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            payout REAL,
            created_at TEXT NOT NULL,
            settled_at TEXT
        );

        CREATE TABLE IF NOT EXISTS favorites (
            chat_id INTEGER NOT NULL,
            team TEXT NOT NULL,
            PRIMARY KEY (chat_id, team)
        );

        CREATE TABLE IF NOT EXISTS market_prefs (
            chat_id INTEGER PRIMARY KEY,
            markets TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS quota (
            chat_id INTEGER NOT NULL,
            week TEXT NOT NULL,
            calls INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (chat_id, week)
        );
        """)


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _iso_week():
    return datetime.now(timezone.utc).strftime("%G-W%V")


# --- Elo ratings ---

def get_elo(team: str, default: float = 1500.0) -> float:
    with _conn() as c:
        row = c.execute("SELECT rating FROM elo_ratings WHERE team = ?", (team,)).fetchone()
        return row["rating"] if row else default


def set_elo(team: str, rating: float):
    with _conn() as c:
        c.execute(
            "INSERT INTO elo_ratings (team, rating) VALUES (?, ?) "
            "ON CONFLICT(team) DO UPDATE SET rating = excluded.rating",
            (team, rating),
        )


# --- Audits / history ---

def record_audit(chat_id, match, home_team, away_team, verdict, summary, markets, full_analysis, prediction: dict):
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO audits (chat_id, match, home_team, away_team, verdict, summary, markets, "
            "full_analysis, prediction_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (chat_id, match, home_team, away_team, verdict, summary, markets, full_analysis,
             json.dumps(prediction), _now()),
        )
        return cur.lastrowid


def get_audit(audit_id: int):
    with _conn() as c:
        row = c.execute("SELECT * FROM audits WHERE id = ?", (audit_id,)).fetchone()
        return dict(row) if row else None


def set_audit_outcome(audit_id: int, outcome: str):
    with _conn() as c:
        c.execute("UPDATE audits SET outcome = ? WHERE id = ?", (outcome, audit_id))


def list_recent_audits(chat_id: int, limit: int = 10):
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM audits WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def audit_stats(chat_id: int):
    """PASS/REJECT hit-rate where an outcome has actually been logged via /result."""
    with _conn() as c:
        rows = c.execute(
            "SELECT verdict, outcome FROM audits WHERE chat_id = ? AND outcome IS NOT NULL",
            (chat_id,),
        ).fetchall()
    pass_win = pass_total = reject_would_have_won = reject_total = 0
    for r in rows:
        if r["verdict"] == "PASS":
            pass_total += 1
            if r["outcome"] == "win":
                pass_win += 1
        elif r["verdict"] == "REJECT":
            reject_total += 1
            if r["outcome"] == "win":
                reject_would_have_won += 1
    return {
        "pass_total": pass_total,
        "pass_win": pass_win,
        "reject_total": reject_total,
        "reject_would_have_won": reject_would_have_won,
    }


# --- Ledger ---

def open_stake(chat_id: int, stake: float):
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO ledger (chat_id, week, stake, status, created_at) VALUES (?,?,?, 'open', ?)",
            (chat_id, _iso_week(), stake, _now()),
        )
        return cur.lastrowid


def get_open_stake(chat_id: int):
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM ledger WHERE chat_id = ? AND status = 'open' ORDER BY id DESC LIMIT 1",
            (chat_id,),
        ).fetchone()
        return dict(row) if row else None


def settle_stake(ledger_id: int, status: str, payout: float):
    with _conn() as c:
        c.execute(
            "UPDATE ledger SET status = ?, payout = ?, settled_at = ? WHERE id = ?",
            (status, payout, _now(), ledger_id),
        )


def list_ledger(chat_id: int, limit: int = 10):
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM ledger WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# --- Favorites ---

def add_favorite(chat_id: int, team: str):
    with _conn() as c:
        c.execute("INSERT OR IGNORE INTO favorites (chat_id, team) VALUES (?, ?)", (chat_id, team))


def remove_favorite(chat_id: int, team: str):
    with _conn() as c:
        c.execute("DELETE FROM favorites WHERE chat_id = ? AND team = ?", (chat_id, team))


def list_favorites(chat_id: int):
    with _conn() as c:
        rows = c.execute("SELECT team FROM favorites WHERE chat_id = ?", (chat_id,)).fetchall()
        return [r["team"] for r in rows]


# --- Market preferences ---

def get_market_prefs(chat_id: int):
    with _conn() as c:
        row = c.execute("SELECT markets FROM market_prefs WHERE chat_id = ?", (chat_id,)).fetchone()
        return json.loads(row["markets"]) if row else list(DEFAULT_MARKETS)


def set_market_prefs(chat_id: int, markets: list):
    with _conn() as c:
        c.execute(
            "INSERT INTO market_prefs (chat_id, markets) VALUES (?, ?) "
            "ON CONFLICT(chat_id) DO UPDATE SET markets = excluded.markets",
            (chat_id, json.dumps(markets)),
        )


# --- Quota guard ---

def check_and_increment_quota(chat_id: int, weekly_cap: int):
    """Returns (allowed, used_after, cap). Increments only if allowed."""
    week = _iso_week()
    with _conn() as c:
        row = c.execute(
            "SELECT calls FROM quota WHERE chat_id = ? AND week = ?", (chat_id, week)
        ).fetchone()
        used = row["calls"] if row else 0
        if used >= weekly_cap:
            return False, used, weekly_cap
        used += 1
        c.execute(
            "INSERT INTO quota (chat_id, week, calls) VALUES (?, ?, ?) "
            "ON CONFLICT(chat_id, week) DO UPDATE SET calls = excluded.calls",
            (chat_id, week, used),
        )
        return True, used, weekly_cap


def get_quota_usage(chat_id: int, weekly_cap: int):
    week = _iso_week()
    with _conn() as c:
        row = c.execute(
            "SELECT calls FROM quota WHERE chat_id = ? AND week = ?", (chat_id, week)
        ).fetchone()
        used = row["calls"] if row else 0
        return used, weekly_cap
