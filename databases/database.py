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
	CREATE TABLE IF NOT EXISTS questionnaires (
		questionnaire_id INTEGER PRIMARY KEY AUTOINCREMENT,
		title            TEXT    NOT NULL,
		description      TEXT,
		created_at       TEXT    NOT NULL,
		is_active        INTEGER NOT NULL DEFAULT 1
	);
	""",
	"""
	CREATE TABLE IF NOT EXISTS questionnaire_questions (
		question_id      INTEGER PRIMARY KEY AUTOINCREMENT,
		questionnaire_id INTEGER NOT NULL,
		question_text    TEXT    NOT NULL,
		question_type    TEXT    NOT NULL CHECK(question_type IN ('text', 'scale', 'multiple_choice', 'yes_no')),
		options          TEXT,
		order_index      INTEGER NOT NULL,
		is_required      INTEGER NOT NULL DEFAULT 1,
		FOREIGN KEY (questionnaire_id) REFERENCES questionnaires(questionnaire_id) ON DELETE CASCADE
	);
	""",
	"""
	CREATE TABLE IF NOT EXISTS questionnaire_responses (
		response_id      INTEGER PRIMARY KEY AUTOINCREMENT,
		user_id          INTEGER NOT NULL,
		questionnaire_id INTEGER NOT NULL,
		question_id      INTEGER NOT NULL,
		response_text    TEXT,
		response_value   REAL,
		answered_at      TEXT    NOT NULL,
		FOREIGN KEY (user_id)          REFERENCES users(user_id) ON DELETE CASCADE,
		FOREIGN KEY (questionnaire_id) REFERENCES questionnaires(questionnaire_id) ON DELETE CASCADE,
		FOREIGN KEY (question_id)      REFERENCES questionnaire_questions(question_id) ON DELETE CASCADE
	);
	""",
	"""
	CREATE TABLE IF NOT EXISTS video_recordings (
		recording_id     INTEGER PRIMARY KEY AUTOINCREMENT,
		user_id          INTEGER NOT NULL,
		recording_type   TEXT    NOT NULL DEFAULT 'emotion_scan',
		file_path        TEXT,
		duration_seconds REAL,
		recorded_at      TEXT    NOT NULL,
		mood_score       REAL,
		primary_emotion  TEXT,
		frame_count      INTEGER,
		metadata         TEXT,
		FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
	);
	""",
	"""
	CREATE TABLE IF NOT EXISTS video_emotion_frames (
		frame_id       INTEGER PRIMARY KEY AUTOINCREMENT,
		recording_id   INTEGER NOT NULL,
		timestamp      REAL    NOT NULL,
		mood_score     REAL    NOT NULL,
		emotions       TEXT    NOT NULL,
		face_count     INTEGER NOT NULL DEFAULT 1,
		FOREIGN KEY (recording_id) REFERENCES video_recordings(recording_id) ON DELETE CASCADE
	);
	""",
	"""
	CREATE TABLE IF NOT EXISTS chat_sessions (
		session_id   INTEGER PRIMARY KEY AUTOINCREMENT,
		user_id      INTEGER NOT NULL,
		started_at   TEXT    NOT NULL,
		ended_at     TEXT,
		mood_score   INTEGER,
		title        TEXT,
		FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
	);
	""",
	"""
	CREATE TABLE IF NOT EXISTS chat_messages (
		message_id   INTEGER PRIMARY KEY AUTOINCREMENT,
		session_id   INTEGER NOT NULL,
		role         TEXT    NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
		content      TEXT    NOT NULL,
		timestamp    TEXT    NOT NULL,
		FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
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
	"""
	CREATE INDEX IF NOT EXISTS idx_questionnaire_responses_user ON questionnaire_responses(user_id, questionnaire_id, answered_at);
	""",
	"""
	CREATE INDEX IF NOT EXISTS idx_video_recordings_user_time ON video_recordings(user_id, recorded_at);
	""",
	"""
	CREATE INDEX IF NOT EXISTS idx_video_emotion_frames_recording ON video_emotion_frames(recording_id, timestamp);
	""",
	"""
	CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_time ON chat_sessions(user_id, started_at);
	""",
	"""
	CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, timestamp);
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


def get_user_mood_history(user_id: int, limit: int = 50) -> list[dict]:
	"""Retrieve mood entries for a specific user."""
	with get_connection() as conn:
		rows = conn.execute(
			"""
			SELECT 
				me.mood_entry_id,
				me.recorded_at,
				ml.score AS mood_score,
				ml.label AS mood_label,
				me.notes
			FROM mood_entries AS me
			JOIN mood_levels AS ml ON ml.mood_level_id = me.mood_level_id
			WHERE me.user_id = ?
			ORDER BY me.recorded_at DESC
			LIMIT ?
			""",
			(user_id, limit)
		).fetchall()
		
		return [
			{
				"id": row[0],
				"timestamp": row[1],
				"mood_score": row[2],
				"mood_label": row[3],
				"notes": row[4]
			}
			for row in rows
		]


def save_video_recording(
	user_id: int,
	duration_seconds: float,
	mood_score: float,
	primary_emotion: str,
	frame_count: int,
	emotion_history: list[dict],
	file_path: str = None,
	metadata: str = None
) -> int:
	"""Save a video recording and its emotion frame data."""
	import json
	from datetime import datetime
	
	with get_connection() as conn:
		# Insert video recording
		cursor = conn.execute(
			"""
			INSERT INTO video_recordings (
				user_id, recording_type, file_path, duration_seconds,
				recorded_at, mood_score, primary_emotion, frame_count, metadata
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
			""",
			(
				user_id, 'emotion_scan', file_path, duration_seconds,
				datetime.now().isoformat(), mood_score, primary_emotion,
				frame_count, metadata
			)
		)
		recording_id = cursor.lastrowid
		
		# Insert emotion frames
		frame_data = []
		for entry in emotion_history:
			emotions_json = json.dumps(entry.get('detections', []))
			frame_data.append((
				recording_id,
				entry['timestamp'],
				entry['mood'],
				emotions_json,
				len(entry.get('detections', []))
			))
		
		conn.executemany(
			"""
			INSERT INTO video_emotion_frames (
				recording_id, timestamp, mood_score, emotions, face_count
			) VALUES (?, ?, ?, ?, ?)
			""",
			frame_data
		)
		
		conn.commit()
		return recording_id


def get_user_video_recordings(user_id: int, limit: int = 20) -> list[dict]:
	"""Retrieve video recordings for a specific user."""
	with get_connection() as conn:
		rows = conn.execute(
			"""
			SELECT 
				recording_id, recorded_at, duration_seconds,
				mood_score, primary_emotion, frame_count
			FROM video_recordings
			WHERE user_id = ?
			ORDER BY recorded_at DESC
			LIMIT ?
			""",
			(user_id, limit)
		).fetchall()
		
		return [
			{
				"recording_id": row[0],
				"timestamp": row[1],
				"duration": row[2],
				"mood_score": row[3],
				"primary_emotion": row[4],
				"frame_count": row[5]
			}
			for row in rows
		]


def get_video_emotion_frames(recording_id: int) -> list[dict]:
	"""Retrieve emotion frame data for a specific recording."""
	import json
	
	with get_connection() as conn:
		rows = conn.execute(
			"""
			SELECT frame_id, timestamp, mood_score, emotions, face_count
			FROM video_emotion_frames
			WHERE recording_id = ?
			ORDER BY timestamp ASC
			""",
			(recording_id,)
		).fetchall()
		
		return [
			{
				"frame_id": row[0],
				"timestamp": row[1],
				"mood_score": row[2],
				"emotions": json.loads(row[3]),
				"face_count": row[4]
			}
			for row in rows
		]


def create_questionnaire(title: str, description: str = None) -> int:
	"""Create a new questionnaire."""
	from datetime import datetime
	
	with get_connection() as conn:
		cursor = conn.execute(
			"INSERT INTO questionnaires (title, description, created_at) VALUES (?, ?, ?)",
			(title, description, datetime.now().isoformat())
		)
		conn.commit()
		return cursor.lastrowid


def add_questionnaire_question(
	questionnaire_id: int,
	question_text: str,
	question_type: str,
	order_index: int,
	options: str = None,
	is_required: bool = True
) -> int:
	"""Add a question to a questionnaire."""
	with get_connection() as conn:
		cursor = conn.execute(
			"""
			INSERT INTO questionnaire_questions (
				questionnaire_id, question_text, question_type,
				options, order_index, is_required
			) VALUES (?, ?, ?, ?, ?, ?)
			""",
			(questionnaire_id, question_text, question_type, options, order_index, 1 if is_required else 0)
		)
		conn.commit()
		return cursor.lastrowid


def get_questionnaire(questionnaire_id: int) -> dict:
	"""Get a questionnaire with all its questions."""
	with get_connection() as conn:
		# Get questionnaire info
		q_row = conn.execute(
			"SELECT title, description, created_at FROM questionnaires WHERE questionnaire_id = ?",
			(questionnaire_id,)
		).fetchone()
		
		if not q_row:
			return None
		
		# Get questions
		questions = conn.execute(
			"""
			SELECT question_id, question_text, question_type, options, order_index, is_required
			FROM questionnaire_questions
			WHERE questionnaire_id = ?
			ORDER BY order_index
			""",
			(questionnaire_id,)
		).fetchall()
		
		return {
			"id": questionnaire_id,
			"title": q_row[0],
			"description": q_row[1],
			"created_at": q_row[2],
			"questions": [
				{
					"id": q[0],
					"text": q[1],
					"type": q[2],
					"options": q[3],
					"order": q[4],
					"required": bool(q[5])
				}
				for q in questions
			]
		}


def save_questionnaire_response(
	user_id: int,
	questionnaire_id: int,
	question_id: int,
	response_text: str = None,
	response_value: float = None
) -> int:
	"""Save a user's response to a questionnaire question."""
	from datetime import datetime
	
	with get_connection() as conn:
		cursor = conn.execute(
			"""
			INSERT INTO questionnaire_responses (
				user_id, questionnaire_id, question_id,
				response_text, response_value, answered_at
			) VALUES (?, ?, ?, ?, ?, ?)
			""",
			(user_id, questionnaire_id, question_id, response_text, response_value, datetime.now().isoformat())
		)
		conn.commit()
		return cursor.lastrowid


def get_user_questionnaire_responses(user_id: int, questionnaire_id: int) -> list[dict]:
	"""Get a user's responses to a specific questionnaire."""
	with get_connection() as conn:
		rows = conn.execute(
			"""
			SELECT 
				qr.response_id, qq.question_text, qq.question_type,
				qr.response_text, qr.response_value, qr.answered_at
			FROM questionnaire_responses qr
			JOIN questionnaire_questions qq ON qr.question_id = qq.question_id
			WHERE qr.user_id = ? AND qr.questionnaire_id = ?
			ORDER BY qq.order_index
			""",
			(user_id, questionnaire_id)
		).fetchall()
		
		return [
			{
				"response_id": row[0],
				"question": row[1],
				"type": row[2],
				"text_response": row[3],
				"value_response": row[4],
				"answered_at": row[5]
			}
			for row in rows
		]


def create_chat_session(user_id: int, title: str = "New Chat") -> int:
	"""Create a new chat session."""
	from datetime import datetime
	
	with get_connection() as conn:
		cursor = conn.execute(
			"INSERT INTO chat_sessions (user_id, started_at, title) VALUES (?, ?, ?)",
			(user_id, datetime.now().isoformat(), title)
		)
		conn.commit()
		return cursor.lastrowid


def save_chat_message(session_id: int, role: str, content: str) -> int:
	"""Save a chat message to the database."""
	from datetime import datetime
	
	with get_connection() as conn:
		cursor = conn.execute(
			"INSERT INTO chat_messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
			(session_id, role, content, datetime.now().isoformat())
		)
		conn.commit()
		return cursor.lastrowid


def get_chat_session_messages(session_id: int) -> list[dict]:
	"""Retrieve all messages for a chat session."""
	with get_connection() as conn:
		rows = conn.execute(
			"""
			SELECT message_id, role, content, timestamp
			FROM chat_messages
			WHERE session_id = ?
			ORDER BY timestamp ASC
			""",
			(session_id,)
		).fetchall()
		
		return [
			{
				"message_id": row[0],
				"role": row[1],
				"content": row[2],
				"timestamp": row[3]
			}
			for row in rows
		]


def get_user_chat_sessions(user_id: int, limit: int = 20) -> list[dict]:
	"""Get all chat sessions for a user."""
	with get_connection() as conn:
		rows = conn.execute(
			"""
			SELECT session_id, title, started_at, ended_at, mood_score
			FROM chat_sessions
			WHERE user_id = ?
			ORDER BY started_at DESC
			LIMIT ?
			""",
			(user_id, limit)
		).fetchall()
		
		return [
			{
				"session_id": row[0],
				"title": row[1],
				"started_at": row[2],
				"ended_at": row[3],
				"mood_score": row[4]
			}
			for row in rows
		]


def update_chat_session_mood(session_id: int, mood_score: int) -> None:
	"""Update the mood score for a chat session."""
	from datetime import datetime
	
	with get_connection() as conn:
		conn.execute(
			"UPDATE chat_sessions SET mood_score = ?, ended_at = ? WHERE session_id = ?",
			(mood_score, datetime.now().isoformat(), session_id)
		)
		conn.commit()


def get_chat_history_for_ollama(session_id: int) -> list[dict]:
	"""Get chat history formatted for Ollama ChatSession."""
	messages = get_chat_session_messages(session_id)
	return [{"role": msg["role"], "content": msg["content"]} for msg in messages]


def get_user_all_questionnaire_responses(user_id: int, limit: int = 10) -> list[dict]:
	"""Get all questionnaire submissions for a user with summary scores."""
	with get_connection() as conn:
		# Get distinct submission times (group by answered_at date)
		rows = conn.execute(
			"""
			SELECT 
				qr.questionnaire_id,
				q.title,
				DATE(qr.answered_at) as submission_date,
				MIN(qr.answered_at) as first_answer_time,
				COUNT(DISTINCT qr.question_id) as questions_answered,
				AVG(qr.response_value) as avg_response_value
			FROM questionnaire_responses qr
			JOIN questionnaires q ON qr.questionnaire_id = q.questionnaire_id
			WHERE qr.user_id = ?
			GROUP BY qr.questionnaire_id, DATE(qr.answered_at)
			ORDER BY first_answer_time DESC
			LIMIT ?
			""",
			(user_id, limit)
		).fetchall()
		
		return [
			{
				"questionnaire_id": row[0],
				"title": row[1],
				"submission_date": row[2],
				"timestamp": row[3],
				"questions_answered": row[4],
				"avg_response_value": row[5]
			}
			for row in rows
		]


def get_questionnaire_submission_details(user_id: int, questionnaire_id: int, submission_date: str) -> list[dict]:
	"""Get detailed responses for a specific questionnaire submission."""
	with get_connection() as conn:
		rows = conn.execute(
			"""
			SELECT 
				qq.question_text,
				qq.question_type,
				qr.response_text,
				qr.response_value,
				qr.answered_at
			FROM questionnaire_responses qr
			JOIN questionnaire_questions qq ON qr.question_id = qq.question_id
			WHERE qr.user_id = ? 
				AND qr.questionnaire_id = ?
				AND DATE(qr.answered_at) = ?
			ORDER BY qq.order_index
			""",
			(user_id, questionnaire_id, submission_date)
		).fetchall()
		
		return [
			{
				"question": row[0],
				"type": row[1],
				"response_text": row[2],
				"response_value": row[3],
				"answered_at": row[4]
			}
			for row in rows
		]
