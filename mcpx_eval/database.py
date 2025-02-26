import sqlite3
import json
from datetime import datetime
from .models import Score, Results, Test

class Database:
    conn: sqlite3.Connection

    def __init__(self, path: str = "eval.db"):
        self.conn = sqlite3.connect(path)

        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tests (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                max_tool_calls INTEGER,
                prompt TEXT NOT NULL,
                prompt_check TEXT NOT NULL,
                UNIQUE(name)
            );
            CREATE TABLE IF NOT EXISTS eval_results (
                id INTEGER PRIMARY KEY,
                t TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                test_name TEXT NOT NULL,
                model TEXT NOT NULL,
                duration REAL NOT NULL,
                output TEXT NOT NULL,
                description TEXT NOT NULL,
                accuracy REAL NOT NULL,
                tool_use REAL NOT NULL,
                tool_calls INT NOT NULL,
                redundant_tool_calls INT NOT NULL DEFAULT 0,
                clarity REAL NOT NULL DEFAULT 0.0,
                helpfulness REAL NOT NULL DEFAULT 0.0, 
                overall REAL NOT NULL,
                hallucination_score REAL NOT NULL DEFAULT 0.0,
                false_claims TEXT NOT NULL DEFAULT '[]',
                tool_analysis TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(test_name) REFERENCES tests(name)
            );
        """
        )
        self.conn.commit()

    def save_score(self, name: str, score: Score, commit=True):
        if name == "":
            return

        self.conn.execute(
            """
                INSERT INTO eval_results (
                    test_name,
                    model,
                    duration,
                    output,
                    description,
                    accuracy,
                    tool_use,
                    tool_calls,
                    redundant_tool_calls,
                    clarity,
                    helpfulness,
                    overall,
                    hallucination_score,
                    false_claims,
                    tool_analysis
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                name,
                score.model,
                score.duration,
                score.llm_output,
                score.description,
                score.accuracy,
                score.tool_use,
                score.tool_calls,
                score.redundant_tool_calls,
                score.clarity,
                score.helpfulness,
                score.overall,
                score.hallucination_score,
                json.dumps(score.false_claims),
                json.dumps(score.tool_analysis),
            ),
        )
        if commit:
            self.conn.commit()

    def save_test(self, test: "Test"):
        self.conn.execute(
            """
            INSERT OR IGNORE INTO tests (name, max_tool_calls, prompt, prompt_check) VALUES (?, ?, ?, ?);
            """,
            (test.name, test.max_tool_calls, test.prompt, test.check),
        )
        self.conn.commit()

    def save_results(self, name: str, results: Results):
        for score in results.scores:
            self.save_score(name, score, commit=False)
        self.conn.commit()

    def average_results(self, name: str) -> Results:
        # ... [keeping the existing average_results implementation]
        pass

    def generate_json_summary(self):
        # ... [keeping the existing generate_json_summary implementation] 
        pass
