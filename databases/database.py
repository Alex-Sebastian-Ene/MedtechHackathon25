"""Normalized SQLite repository for accounts, mood tracking, and GPS data."""

from __future__ import annotations

import sqlite3
from pprint import pprint
from datetime import datetime
from pathlib import Path
from typing import Iterable

DB_PATH = Path(__file__).with_name("medtech.db")


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
	conn = sqlite3.connect(db_path)
	conn.execute("PRAGMA foreign_keys = ON;")
	return conn


SCHEMA_STATEMENTS: Iterable[str] = (
	"""
	CREATE TABLE IF NOT EXISTS users (
		user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
		username      TEXT    NOT NULL UNIQUE,
		password_hash TEXT    NOT NULL,
		email         TEXT,
		created_at    TEXT    NOT NULL
	);
	""",
	"""
	CREATE TABLE IF NOT EXISTS gps_sessions (
		session_id INTEGER PRIMARY KEY AUTOINCREMENT,
		user_id    INTEGER NOT NULL,
		started_at TEXT    NOT NULL,
		ended_at   TEXT,
		label      TEXT,
		FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
	);
	""",
	"""
	CREATE TABLE IF NOT EXISTS gps_points (
		point_id    INTEGER PRIMARY KEY AUTOINCREMENT,
		session_id  INTEGER NOT NULL,
		recorded_at TEXT    NOT NULL,
		latitude    REAL    NOT NULL,
		longitude   REAL    NOT NULL,
		accuracy_m  REAL,
		FOREIGN KEY (session_id) REFERENCES gps_sessions(session_id) ON DELETE CASCADE
	);
	""",
	"""
	CREATE TABLE IF NOT EXISTS text_entries (
		text_entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
		user_id       INTEGER NOT NULL,
		source        TEXT    NOT NULL DEFAULT 'manual',
		raw_text      TEXT    NOT NULL,
		processed_text TEXT,
		language      TEXT,
		created_at    TEXT    NOT NULL,
		FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
	);
	""",
	"""
	CREATE TABLE IF NOT EXISTS locations (
		location_id INTEGER PRIMARY KEY AUTOINCREMENT,
		user_id     INTEGER NOT NULL,
		label       TEXT    NOT NULL CHECK(label IN ('home', 'work')),
		latitude    REAL    NOT NULL,
		longitude   REAL    NOT NULL,
		radius_m    REAL    NOT NULL DEFAULT 50,
		created_at  TEXT    NOT NULL,
		FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
		UNIQUE (user_id, label)
	);
	""",
	"""
	CREATE TABLE IF NOT EXISTS mood_levels (
		mood_level_id INTEGER PRIMARY KEY,
		label         TEXT    NOT NULL UNIQUE,
		score         INTEGER NOT NULL CHECK(score BETWEEN 1 AND 10)
	);
	""",
	"""
	CREATE TABLE IF NOT EXISTS mood_entries (
		mood_entry_id  INTEGER PRIMARY KEY AUTOINCREMENT,
		user_id        INTEGER NOT NULL,
		mood_level_id  INTEGER NOT NULL,
		text_entry_id  INTEGER,
		recorded_at    TEXT    NOT NULL,
		notes          TEXT,
		FOREIGN KEY (user_id)       REFERENCES users(user_id) ON DELETE CASCADE,
		FOREIGN KEY (mood_level_id) REFERENCES mood_levels(mood_level_id) ON UPDATE CASCADE,
		FOREIGN KEY (text_entry_id) REFERENCES text_entries(text_entry_id) ON DELETE SET NULL
	);
	""",
	"""
	CREATE INDEX IF NOT EXISTS idx_gps_points_session_time ON gps_points(session_id, recorded_at);
	""",
	"""
	CREATE INDEX IF NOT EXISTS idx_text_entries_user_time ON text_entries(user_id, created_at);
	""",
	"""
	CREATE INDEX IF NOT EXISTS idx_mood_entries_user_time ON mood_entries(user_id, recorded_at);
	""",
)


MOOD_LEVEL_ROWS = tuple((score, f"level_{score}", score) for score in range(1, 11))

LOCATION_LABELS = ("home", "work")

MOOD_SUMMARY_QUERY = """
SELECT
	me.mood_entry_id,
	u.username,
	me.recorded_at,
	ml.score AS mood_score,
	ml.label AS mood_label,
	me.notes
FROM mood_entries AS me
JOIN users AS u ON u.user_id = me.user_id
JOIN mood_levels AS ml ON ml.mood_level_id = me.mood_level_id
ORDER BY me.recorded_at DESC
LIMIT 20;
"""

#from databases.database import get_connection, initialize_database, MOOD_SUMMARY_QUERY

# initialize_database()  # makes sure schema + mood levels exist
# with get_connection() as conn:
#     rows = conn.execute(MOOD_SUMMARY_QUERY).fetchall()
#     for row in rows:
#         print(row)

def initialize_schema(conn: sqlite3.Connection) -> None:
	for statement in SCHEMA_STATEMENTS:
		conn.executescript(statement)


def seed_reference_data(conn: sqlite3.Connection) -> None:
	existing = conn.execute("SELECT COUNT(*) FROM mood_levels;").fetchone()[0]
	if existing:
		return
	conn.executemany(
		"INSERT INTO mood_levels (mood_level_id, label, score) VALUES (?, ?, ?);",
		MOOD_LEVEL_ROWS,
	)


def initialize_database(db_path: Path = DB_PATH) -> Path:
	db_path.parent.mkdir(parents=True, exist_ok=True)
	with get_connection(db_path) as conn:
		initialize_schema(conn)
		seed_reference_data(conn)
		conn.commit()
	return db_path


def describe_tables(conn: sqlite3.Connection) -> dict[str, list[tuple[str, str, str]]]:
	description: dict[str, list[tuple[str, str, str]]] = {}
	tables = conn.execute(
		"SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
	).fetchall()
	for (table_name,) in tables:
		columns = conn.execute(f"PRAGMA table_info({table_name});").fetchall()
		description[table_name] = [(col[1], col[2], "NOT NULL" if col[3] else "") for col in columns]
	return description
