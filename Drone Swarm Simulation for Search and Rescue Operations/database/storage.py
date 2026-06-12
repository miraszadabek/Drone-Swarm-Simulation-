import sqlite3
import time
from config.settings import DB_PATH

class MissionStorage:
    def __init__(self):
        self.db_path = DB_PATH
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    drone_id INTEGER,
                    action TEXT,
                    reasoning TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mission_summaries (
                    mission_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    coverage_pct REAL,
                    victims_found INTEGER,
                    total_llm_calls INTEGER,
                    elapsed_seconds REAL
                )
            """)
            conn.commit()

    def save_decision(self, drone_id: int, action: str, reasoning: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO ai_decisions (timestamp, drone_id, action, reasoning) VALUES (?, ?, ?, ?)",
                (time.time(), drone_id, action, reasoning)
            )
            conn.commit()

    def save_mission_summary(self, coverage: float, found: int, llm_calls: int, elapsed: float):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO mission_summaries (timestamp, coverage_pct, victims_found, total_llm_calls, elapsed_seconds) VALUES (?, ?, ?, ?, ?)",
                (time.time(), coverage, found, llm_calls, elapsed)
            )
            conn.commit()